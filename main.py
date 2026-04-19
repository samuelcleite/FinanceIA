"""
FinanceIA RAG API — API intermediária entre Base44 e Supabase/pgvector.

Fluxo:
1. Base44 envia pergunta do usuário (+ filtros opcionais)
2. API expande a query via Haiku
3. API gera embedding via OpenAI
4. API busca fundos ampla no Supabase (top_k alto)
5. API enriquece com dados quantitativos
6. API aplica reranking quantitativo (rentabilidade, volatilidade, PL, consistência, captação)
7. API retorna os melhores fundos formatados para o Claude

Hospedagem: Railway
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import logging
import math
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from supabase import create_client, Client

# ──────────────────────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financeia-rag")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "dev-key-trocar-em-producao")

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536

# ── Reranking: busca semântica ampla, depois filtra por qualidade ──
# Quantos fundos buscar do pgvector antes do reranking
RERANK_FETCH_K = 50
# Similaridade mínima para considerar (abaixo disso, descarta)
RERANK_MIN_SIMILARITY = 0.25

# Pesos do scoring quantitativo (somam 1.0)
# Ordem de prioridade: Rentabilidade > Volatilidade > PL > Consistência > Captação
RERANK_WEIGHTS = {
    "rentabilidade": 0.35,
    "volatilidade": 0.25,
    "pl": 0.20,
    "consistencia": 0.12,
    "captacao": 0.08,
}

# ──────────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────────

openai_client: Optional[OpenAI] = None
supabase_client: Optional[Client] = None

# ──────────────────────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinanceIA RAG API",
    description="API intermediária para busca semântica de fundos de investimento",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Autenticação
# ──────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not api_key or api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="API key inválida ou ausente")
    return api_key


# ──────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    global openai_client, supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("⚠️  SUPABASE_URL ou SUPABASE_SERVICE_KEY não configurados")
    if not OPENAI_API_KEY:
        logger.warning("⚠️  OPENAI_API_KEY não configurada")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Clients inicializados")


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────


class BuscaRequest(BaseModel):
    pergunta: str = Field(..., description="Pergunta do usuário sobre fundos")
    plataformas: Optional[list[str]] = Field(None, description="Filtrar por plataformas")
    plataforma: Optional[str] = Field(None, description="Filtrar por plataforma única (retrocompatível)")
    categoria: Optional[str] = Field(None, description="Filtrar por categoria")
    tipo_produto: Optional[str] = Field(None, description="Filtrar por tipo de produto")
    top_k: int = Field(5, ge=1, le=20, description="Número de fundos a retornar ao Claude")


class FundoContexto(BaseModel):
    cnpj: str
    nome: str
    similaridade: float
    score_quantitativo: float = 0.0
    score_final: float = 0.0
    contexto_texto: str
    dados_estruturados: dict


class BuscaResponse(BaseModel):
    pergunta_original: str
    filtros_aplicados: dict
    total_candidatos: int  # quantos passaram pela busca semântica
    total_resultados: int  # quantos foram retornados após reranking
    fundos: list[FundoContexto]
    contexto_consolidado: str


class Premissa(BaseModel):
    id: int
    titulo: str
    categoria: Optional[str] = None
    conteudo: str
    atualizado_em: Optional[str] = None


class PremissasResponse(BaseModel):
    total: int
    premissas: list[Premissa]
    contexto_premissas: str


# ──────────────────────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────────────────────


def gerar_embedding(texto: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def expandir_query(pergunta: str) -> str:
    EXPANSION_PROMPT = """Você é um assistente que reescreve perguntas de clientes sobre investimentos
para melhorar a busca semântica em um banco de dados de fundos de investimento.

O banco contém fundos com campos como: nome, gestor, categoria, subcategoria, tipo_produto,
indexador, benchmark, tributacao, come_cotas, descricao_tecnica, perfil_recomendado,
riscos_e_restricoes.

Reescreva a pergunta adicionando sinônimos técnicos e termos relacionados que provavelmente
aparecem nos documentos dos fundos. Mantenha a pergunta original e adicione os termos extras.

IMPORTANTE: retorne APENAS a query expandida, sem explicação.

Exemplos:
- "isento de IR" → "renda fixa isento IR isenção imposto de renda debêntures incentivadas infraestrutura lei 12431 tributação isento pessoa física"
- "fundo conservador" → "fundo conservador baixo risco renda fixa CDI pós-fixado liquidez diária volatilidade baixa"
- "proteção contra inflação" → "proteção inflação IPCA indexado inflação NTN-B tesouro IPCA+ hedge inflacionário"
- "fundo para aposentadoria" → "fundo aposentadoria previdência longo prazo PGBL VGBL previdenciário"
"""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": f"Pergunta do cliente: {pergunta}"}],
            system=EXPANSION_PROMPT,
        )
        expanded = response.content[0].text.strip()
        logger.info(f"   Query expandida: {expanded[:100]}...")
        return expanded
    except Exception as e:
        logger.warning(f"   Query expansion falhou ({e}), usando query original")
        return pergunta


def buscar_fundos_supabase(
    query_embedding: list[float],
    plataforma: Optional[str] = None,
    categoria: Optional[str] = None,
    tipo_produto: Optional[str] = None,
    top_k: int = 50,
) -> list[dict]:
    """Chama a função RPC buscar_fundos no Supabase."""
    params = {
        "query_embedding": query_embedding,
        "top_k": top_k,
    }

    if plataforma:
        params["filtro_plataforma"] = plataforma
    if categoria:
        params["filtro_categoria"] = categoria
    if tipo_produto:
        params["filtro_tipo_produto"] = tipo_produto

    result = supabase_client.rpc("buscar_fundos", params).execute()
    return result.data or []


def enriquecer_com_dados_complementares(fundos: list[dict]) -> list[dict]:
    """
    Busca dados complementares:
    - Plataformas disponíveis (fundo_plataformas)
    - Rentabilidade, PL, volatilidade, meses negativos, captação (fundo_infos_atualizadas)
    - Campos extras da tabela fundos (taxa_adm, taxa_performance, publico_alvo)
    """
    if not fundos:
        return fundos

    cnpjs = [f["cnpj"] for f in fundos]

    # 1. Buscar plataformas
    try:
        plats_result = (
            supabase_client.table("fundo_plataformas")
            .select("cnpj, plataforma")
            .in_("cnpj", cnpjs)
            .eq("disponivel", True)
            .execute()
        )
        plats_por_cnpj = {}
        for p in (plats_result.data or []):
            plats_por_cnpj.setdefault(p["cnpj"], []).append(p["plataforma"])
    except Exception as e:
        logger.warning(f"Erro ao buscar plataformas: {e}")
        plats_por_cnpj = {}

    # 2. Buscar dados atualizados (rentabilidade, PL, volatilidade, meses negativos, captação)
    try:
        infos_result = (
            supabase_client.table("fundo_infos_atualizadas")
            .select("*")
            .in_("cnpj", cnpjs)
            .execute()
        )
        infos_por_cnpj = {i["cnpj"]: i for i in (infos_result.data or [])}
    except Exception as e:
        logger.warning(f"Erro ao buscar infos atualizadas: {e}")
        infos_por_cnpj = {}

    # 3. Buscar campos extras da tabela fundos
    try:
        extras_result = (
            supabase_client.table("fundos")
            .select("cnpj, taxa_adm, taxa_performance, publico_alvo")
            .in_("cnpj", cnpjs)
            .execute()
        )
        extras_por_cnpj = {e["cnpj"]: e for e in (extras_result.data or [])}
    except Exception as e:
        logger.warning(f"Erro ao buscar campos extras: {e}")
        extras_por_cnpj = {}

    # 4. Mesclar tudo
    for fundo in fundos:
        cnpj = fundo["cnpj"]
        fundo["plataformas"] = plats_por_cnpj.get(cnpj, [])

        infos = infos_por_cnpj.get(cnpj, {})
        for campo in [
            "retorno_12m", "retorno_24m", "retorno_36m", "retorno_inicio",
            "pl_atual", "vol_12m", "meses_neg_12m", "captacao_liquida_30d",
        ]:
            if infos.get(campo) is not None:
                fundo[campo] = infos[campo]

        extras = extras_por_cnpj.get(cnpj, {})
        for campo in ["taxa_adm", "taxa_performance", "publico_alvo"]:
            if extras.get(campo) is not None:
                fundo[campo] = extras[campo]

    return fundos


# ──────────────────────────────────────────────────────────────
# Reranking quantitativo
# ──────────────────────────────────────────────────────────────


def normalizar_min_max(valores: list[Optional[float]], inverter: bool = False) -> list[float]:
    """
    Normaliza uma lista de valores para [0, 1] usando min-max.
    Se inverter=True, valores menores recebem score maior (para volatilidade, meses negativos).
    Valores None recebem 0.0 (penalização por falta de dados).
    """
    nums = [v for v in valores if v is not None]
    if not nums or max(nums) == min(nums):
        return [0.5 if v is not None else 0.0 for v in valores]

    vmin, vmax = min(nums), max(nums)
    resultado = []
    for v in valores:
        if v is None:
            resultado.append(0.0)
        else:
            norm = (v - vmin) / (vmax - vmin)
            resultado.append(1.0 - norm if inverter else norm)
    return resultado


def calcular_score_rentabilidade(fundo: dict) -> Optional[float]:
    """
    Score de rentabilidade: média ponderada de 12m (peso 3), 24m (peso 2), 36m (peso 1).
    Prioriza performance recente.
    """
    r12 = fundo.get("retorno_12m")
    r24 = fundo.get("retorno_24m")
    r36 = fundo.get("retorno_36m")

    pesos = []
    valores = []
    if r12 is not None:
        valores.append(r12)
        pesos.append(3)
    if r24 is not None:
        valores.append(r24)
        pesos.append(2)
    if r36 is not None:
        valores.append(r36)
        pesos.append(1)

    if not valores:
        return None

    return sum(v * p for v, p in zip(valores, pesos)) / sum(pesos)


def reranking_quantitativo(fundos: list[dict], top_k: int) -> list[dict]:
    """
    Aplica scoring quantitativo aos fundos já enriquecidos com dados da fundo_infos_atualizadas.

    Critérios (em ordem de peso):
    1. Rentabilidade (35%) — média ponderada 12m/24m/36m
    2. Volatilidade baixa (25%) — invertido: menor vol = melhor score
    3. Tamanho do fundo PL (20%) — maior PL = mais score
    4. Consistência (12%) — meses negativos invertido: menos meses neg = melhor
    5. Captação líquida (8%) — maior captação = mais confiança do mercado

    Fundos sem dados quantitativos recebem score 0 nos critérios faltantes,
    o que os penaliza naturalmente no ranking.
    """
    if not fundos:
        return fundos

    n = len(fundos)
    logger.info(f"   📊 Reranking: {n} candidatos → top {top_k}")

    # Extrair valores brutos para cada critério
    rent_brutos = [calcular_score_rentabilidade(f) for f in fundos]
    vol_brutos = [f.get("vol_12m") for f in fundos]
    pl_brutos = [f.get("pl_atual") for f in fundos]
    consist_brutos = [f.get("meses_neg_12m") for f in fundos]
    capt_brutos = [f.get("captacao_liquida_30d") for f in fundos]

    # Normalizar para [0, 1]
    rent_norm = normalizar_min_max(rent_brutos, inverter=False)
    vol_norm = normalizar_min_max(vol_brutos, inverter=True)       # menor vol = melhor
    pl_norm = normalizar_min_max(pl_brutos, inverter=False)
    consist_norm = normalizar_min_max(consist_brutos, inverter=True)  # menos meses neg = melhor
    capt_norm = normalizar_min_max(capt_brutos, inverter=False)

    # Calcular score composto para cada fundo
    w = RERANK_WEIGHTS
    for i, fundo in enumerate(fundos):
        score_quant = (
            w["rentabilidade"] * rent_norm[i]
            + w["volatilidade"] * vol_norm[i]
            + w["pl"] * pl_norm[i]
            + w["consistencia"] * consist_norm[i]
            + w["captacao"] * capt_norm[i]
        )
        fundo["_score_quantitativo"] = round(score_quant, 4)
        fundo["_score_similaridade"] = fundo.get("similaridade", 0)

        # Score final: 40% semântico + 60% quantitativo
        # A semântica garante relevância temática, o quantitativo garante qualidade
        sim_normalizado = fundo.get("similaridade", 0)
        fundo["_score_final"] = round(0.4 * sim_normalizado + 0.6 * score_quant, 4)

    # Ordenar pelo score final (maior = melhor)
    fundos.sort(key=lambda f: f["_score_final"], reverse=True)

    # Log dos top resultados para debug
    for i, f in enumerate(fundos[:top_k]):
        logger.info(
            f"   #{i+1} {f.get('nome', '?')[:50]} | "
            f"sim={f['_score_similaridade']:.4f} | "
            f"quant={f['_score_quantitativo']:.4f} | "
            f"final={f['_score_final']:.4f} | "
            f"r12m={f.get('retorno_12m', 'N/A')} | "
            f"vol={f.get('vol_12m', 'N/A')} | "
            f"pl={f.get('pl_atual', 'N/A')}"
        )

    return fundos[:top_k]


# ──────────────────────────────────────────────────────────────
# Formatação
# ──────────────────────────────────────────────────────────────


def formatar_fundo_contexto(fundo: dict) -> FundoContexto:
    """Transforma o resultado da busca em contexto formatado para o prompt do Claude."""

    linhas = []
    linhas.append(f"📌 {fundo.get('nome', 'N/A')} (CNPJ: {fundo.get('cnpj', 'N/A')})")

    if fundo.get("gestor"):
        linhas.append(f"   Gestor: {fundo['gestor']}")

    cats = [c for c in [fundo.get("tipo_produto"), fundo.get("categoria"), fundo.get("subcategoria")] if c]
    if cats:
        linhas.append(f"   Classificação: {' > '.join(cats)}")

    if fundo.get("indexador"):
        linhas.append(f"   Indexador: {fundo['indexador']}")
    if fundo.get("benchmark"):
        linhas.append(f"   Benchmark: {fundo['benchmark']}")
    if fundo.get("tributacao"):
        linhas.append(f"   Tributação: {fundo['tributacao']}")
    if fundo.get("come_cotas"):
        linhas.append(f"   Come-cotas: {fundo['come_cotas']}")

    taxas = []
    if fundo.get("taxa_adm") is not None:
        taxas.append(f"Adm: {fundo['taxa_adm']}%")
    if fundo.get("taxa_performance"):
        taxas.append(f"Perf: {fundo['taxa_performance']}")
    if taxas:
        linhas.append(f"   Taxas: {' | '.join(taxas)}")

    if fundo.get("liquidez_dias"):
        linhas.append(f"   Liquidez: {fundo['liquidez_dias']} dias")
    if fundo.get("horizonte_minimo_anos") is not None:
        linhas.append(f"   Horizonte mínimo: {fundo['horizonte_minimo_anos']} anos")
    if fundo.get("aplicacao_minima") is not None:
        linhas.append(f"   Aplicação mínima: R$ {fundo['aplicacao_minima']:,.2f}")
    if fundo.get("publico_alvo"):
        linhas.append(f"   Público-alvo: {fundo['publico_alvo']}")

    # Campos qualitativos
    if fundo.get("descricao_tecnica"):
        linhas.append(f"   Descrição técnica: {fundo['descricao_tecnica']}")
    if fundo.get("perfil_recomendado"):
        linhas.append(f"   Perfil recomendado: {fundo['perfil_recomendado']}")
    if fundo.get("riscos_e_restricoes"):
        linhas.append(f"   Riscos e restrições: {fundo['riscos_e_restricoes']}")

    # Dados quantitativos
    rents = []
    for periodo, label in [("retorno_12m", "12m"), ("retorno_24m", "24m"), ("retorno_36m", "36m")]:
        val = fundo.get(periodo)
        if val is not None:
            rents.append(f"{label}: {val:.2f}%")
    if rents:
        linhas.append(f"   Rentabilidade: {' | '.join(rents)}")

    if fundo.get("vol_12m") is not None:
        linhas.append(f"   Volatilidade 12m: {fundo['vol_12m']:.2f}%")
    if fundo.get("meses_neg_12m") is not None:
        linhas.append(f"   Meses negativos (12m): {fundo['meses_neg_12m']}")
    if fundo.get("pl_atual") is not None:
        linhas.append(f"   PL atual: R$ {fundo['pl_atual']:,.0f}")
    if fundo.get("captacao_liquida_30d") is not None:
        linhas.append(f"   Captação líquida 30d: R$ {fundo['captacao_liquida_30d']:,.0f}")

    # Plataformas
    if fundo.get("plataformas"):
        linhas.append(f"   Disponível em: {', '.join(fundo['plataformas'])}")

    contexto_texto = "\n".join(linhas)
    similaridade = fundo.get("similaridade", 0)

    return FundoContexto(
        cnpj=fundo.get("cnpj", ""),
        nome=fundo.get("nome", "N/A"),
        similaridade=round(similaridade, 4),
        score_quantitativo=round(fundo.get("_score_quantitativo", 0), 4),
        score_final=round(fundo.get("_score_final", 0), 4),
        contexto_texto=contexto_texto,
        dados_estruturados={
            k: v
            for k, v in fundo.items()
            if k not in ("embedding", "documento_texto") and not k.startswith("_") and v is not None
        },
    )


def montar_contexto_consolidado(
    fundos_formatados: list[FundoContexto],
    pergunta: str,
    filtros: dict,
) -> str:
    header = "=" * 60
    linhas = [
        header,
        "CONTEXTO: FUNDOS DE INVESTIMENTO RELEVANTES",
        "(Ordenados por relevância semântica + qualidade quantitativa)",
        header,
        f"Pergunta do cliente: {pergunta}",
    ]

    filtros_ativos = {k: v for k, v in filtros.items() if v}
    if filtros_ativos:
        linhas.append(f"Filtros aplicados: {json.dumps(filtros_ativos, ensure_ascii=False)}")

    linhas.append(f"Total de fundos selecionados: {len(fundos_formatados)}")
    linhas.append(header)

    for i, fundo in enumerate(fundos_formatados, 1):
        linhas.append(f"\n--- Fundo {i}/{len(fundos_formatados)} (relevância: {fundo.score_final:.2f}) ---")
        linhas.append(fundo.contexto_texto)

    linhas.append(f"\n{header}")
    linhas.append("FIM DO CONTEXTO")
    linhas.append(header)

    return "\n".join(linhas)


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "financeia-rag-api",
        "version": "2.0.0",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "rerank_fetch_k": RERANK_FETCH_K,
        "rerank_weights": RERANK_WEIGHTS,
    }


@app.post("/buscar", response_model=BuscaResponse)
async def buscar_fundos(
    request: BuscaRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Endpoint principal com reranking quantitativo.

    Fluxo:
    1. Expande a query (Haiku)
    2. Gera embedding (OpenAI)
    3. Busca semântica ampla (top 50 do pgvector)
    4. Filtra por similaridade mínima
    5. Enriquece com dados quantitativos
    6. Reranking: 40% semântico + 60% quantitativo
    7. Retorna top_k melhores formatados
    """
    logger.info(
        f"🔍 Busca: '{request.pergunta}' | plataforma={request.plataforma} | "
        f"plataformas={request.plataformas} | categoria={request.categoria} | "
        f"top_k_final={request.top_k}"
    )

    try:
        # 1. Expandir a query
        query_expandida = expandir_query(request.pergunta)

        # 2. Gerar embedding
        embedding = gerar_embedding(query_expandida)
        logger.info(f"   Embedding gerado ({len(embedding)} dimensões)")

    except Exception as e:
        logger.error(f"   ❌ Erro ao gerar embedding: {e}")
        raise HTTPException(status_code=502, detail=f"Erro ao gerar embedding: {str(e)}")

    try:
        # 3. Busca semântica ampla — pega RERANK_FETCH_K fundos
        plataformas_lista = request.plataformas or ([request.plataforma] if request.plataforma else [None])

        todos_resultados = {}
        for plat in plataformas_lista:
            resultados_plat = buscar_fundos_supabase(
                query_embedding=embedding,
                plataforma=plat,
                categoria=request.categoria,
                tipo_produto=request.tipo_produto,
                top_k=RERANK_FETCH_K,
            )
            for r in resultados_plat:
                cnpj = r.get("cnpj")
                if cnpj not in todos_resultados:
                    todos_resultados[cnpj] = r

        # 4. Filtrar por similaridade mínima
        candidatos = [
            f for f in todos_resultados.values()
            if f.get("similaridade", 0) >= RERANK_MIN_SIMILARITY
        ]

        # Ordenar por similaridade para log
        candidatos.sort(key=lambda x: x.get("similaridade", 0), reverse=True)

        logger.info(
            f"   {len(todos_resultados)} fundos do pgvector → "
            f"{len(candidatos)} acima do threshold ({RERANK_MIN_SIMILARITY})"
        )

    except Exception as e:
        logger.error(f"   ❌ Erro na busca Supabase: {e}")
        raise HTTPException(status_code=502, detail=f"Erro na busca vetorial: {str(e)}")

    # 5. Enriquecer com dados quantitativos
    candidatos = enriquecer_com_dados_complementares(candidatos)

    # 6. Reranking quantitativo → retorna top_k
    melhores = reranking_quantitativo(candidatos, request.top_k)

    # 7. Formatar
    fundos_formatados = [formatar_fundo_contexto(f) for f in melhores]

    filtros = {
        "plataforma": request.plataforma,
        "plataformas": request.plataformas,
        "categoria": request.categoria,
        "tipo_produto": request.tipo_produto,
    }

    contexto = montar_contexto_consolidado(fundos_formatados, request.pergunta, filtros)

    return BuscaResponse(
        pergunta_original=request.pergunta,
        filtros_aplicados=filtros,
        total_candidatos=len(candidatos),
        total_resultados=len(fundos_formatados),
        fundos=fundos_formatados,
        contexto_consolidado=contexto,
    )


@app.post("/contexto-completo")
async def contexto_completo(
    request: BuscaRequest,
    _api_key: str = Depends(verify_api_key),
):
    resultado = await buscar_fundos(request, _api_key)
    return {"contexto": resultado.contexto_consolidado}


@app.get("/stats")
async def stats(_api_key: str = Depends(verify_api_key)):
    try:
        total = supabase_client.table("fundos").select("cnpj", count="exact").execute()
        embeddings = supabase_client.table("fundo_embeddings").select("cnpj", count="exact").execute()
        plataformas = (
            supabase_client.table("fundo_plataformas")
            .select("plataforma")
            .execute()
        )
        plats_unicas = list(set(p["plataforma"] for p in (plataformas.data or [])))

        return {
            "total_fundos": total.count,
            "total_embeddings": embeddings.count,
            "plataformas": sorted(plats_unicas),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
            "rerank_fetch_k": RERANK_FETCH_K,
            "rerank_min_similarity": RERANK_MIN_SIMILARITY,
            "rerank_weights": RERANK_WEIGHTS,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/premissas", response_model=PremissasResponse)
async def listar_premissas(
    categoria: Optional[str] = None,
    _api_key: str = Depends(verify_api_key),
):
    try:
        query = (
            supabase_client.table("premissas")
            .select("*")
            .eq("ativo", True)
            .order("categoria")
            .order("titulo")
        )

        if categoria:
            query = query.eq("categoria", categoria)

        result = query.execute()
        dados = result.data or []

    except Exception as e:
        logger.error(f"❌ Erro ao buscar premissas: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar premissas: {str(e)}")

    premissas = [
        Premissa(
            id=p["id"],
            titulo=p["titulo"],
            categoria=p.get("categoria"),
            conteudo=p["conteudo"],
            atualizado_em=str(p.get("atualizado_em", "")),
        )
        for p in dados
    ]

    contexto = formatar_premissas_para_prompt(premissas)

    return PremissasResponse(
        total=len(premissas),
        premissas=premissas,
        contexto_premissas=contexto,
    )


@app.put("/premissas/{premissa_id}")
async def atualizar_premissa(
    premissa_id: int,
    dados: dict,
    _api_key: str = Depends(verify_api_key),
):
    campos_permitidos = {"titulo", "categoria", "conteudo", "ativo"}
    update_data = {k: v for k, v in dados.items() if k in campos_permitidos}

    if not update_data:
        raise HTTPException(status_code=400, detail="Nenhum campo válido para atualizar")

    try:
        result = (
            supabase_client.table("premissas")
            .update(update_data)
            .eq("id", premissa_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Premissa não encontrada")

        return {"status": "ok", "premissa": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def formatar_premissas_para_prompt(premissas: list[Premissa]) -> str:
    if not premissas:
        return ""

    linhas = [
        "=" * 60,
        "PREMISSAS E DIRETRIZES DO ASSESSOR",
        "Considere SEMPRE as premissas abaixo ao elaborar suas respostas.",
        "=" * 60,
    ]

    por_categoria = {}
    for p in premissas:
        cat = p.categoria or "geral"
        por_categoria.setdefault(cat, []).append(p)

    ordem = ["macro", "planejamento", "alocacao", "regras", "alertas", "geral"]
    categorias_ordenadas = sorted(
        por_categoria.keys(),
        key=lambda c: ordem.index(c) if c in ordem else 99,
    )

    for cat in categorias_ordenadas:
        for p in por_categoria[cat]:
            linhas.append(f"\n=== {p.titulo} [{cat.upper()}] ===")
            linhas.append(p.conteudo)

    linhas.append(f"\n{'=' * 60}")
    linhas.append("FIM DAS PREMISSAS")
    linhas.append("=" * 60)

    return "\n".join(linhas)


# ──────────────────────────────────────────────────────────────
# Execução local
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)