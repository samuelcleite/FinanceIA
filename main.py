"""
FinanceIA RAG API — API intermediária entre Base44 e Supabase/pgvector.

Fluxo:
1. Base44 envia pergunta do usuário (+ filtros opcionais)
2. API gera embedding via OpenAI
3. API chama função buscar_fundos no Supabase (RPC)
4. API formata o contexto dos fundos retornados
5. API retorna contexto formatado para o Base44 montar o prompt e chamar Claude

Hospedagem recomendada: Railway, Render ou Fly.io (free tier suficiente).
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import logging
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

# Variáveis de ambiente (configurar no deploy)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "dev-key-trocar-em-producao")

# Modelo de embedding — mesmo usado na geração dos vetores
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536

# ──────────────────────────────────────────────────────────────
# Clients (inicializados no startup)
# ──────────────────────────────────────────────────────────────

openai_client: Optional[OpenAI] = None
supabase_client: Optional[Client] = None

# ──────────────────────────────────────────────────────────────
# App FastAPI
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinanceIA RAG API",
    description="API intermediária para busca semântica de fundos de investimento",
    version="1.0.0",
)

# CORS — liberar para o domínio do Base44
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir ao domínio do Base44
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Autenticação simples via API Key
# ──────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not api_key or api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="API key inválida ou ausente")
    return api_key


# ──────────────────────────────────────────────────────────────
# Startup / Shutdown
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
# Schemas (Pydantic)
# ──────────────────────────────────────────────────────────────


class BuscaRequest(BaseModel):
    """Payload enviado pelo Base44."""

    pergunta: str = Field(..., description="Pergunta do usuário sobre fundos")
    plataformas: Optional[list[str]] = Field(
        None,
        description="Filtrar por plataformas (ex: ['XP', 'BTG']). Aceita múltiplas.",
    )
    plataforma: Optional[str] = Field(
        None,
        description="Filtrar por plataforma única (retrocompatível). Use 'plataformas' para múltiplas.",
    )
    categoria: Optional[str] = Field(
        None,
        description="Filtrar por categoria (ex: 'Renda Fixa', 'Multimercado')",
    )
    tipo_produto: Optional[str] = Field(
        None,
        description="Filtrar por tipo de produto (ex: 'Fundos', 'Previdência')",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Número de fundos a retornar (padrão: 5)",
    )


class FundoContexto(BaseModel):
    """Um fundo retornado pela busca, já formatado para contexto."""

    cnpj: str
    nome: str
    similaridade: float
    contexto_texto: str  # Texto formatado para incluir no prompt do Claude
    dados_estruturados: dict  # Dados brutos para o Base44 usar se precisar


class BuscaResponse(BaseModel):
    """Resposta da API com os fundos encontrados."""

    pergunta_original: str
    filtros_aplicados: dict
    total_resultados: int
    fundos: list[FundoContexto]
    contexto_consolidado: str  # Texto pronto para colar no prompt do Claude


class Premissa(BaseModel):
    """Uma premissa/guidance retornada pela API."""

    id: int
    titulo: str
    categoria: Optional[str] = None
    conteudo: str
    atualizado_em: Optional[str] = None


class PremissasResponse(BaseModel):
    """Resposta com todas as premissas ativas, formatadas para o prompt."""

    total: int
    premissas: list[Premissa]
    contexto_premissas: str  # Texto pronto para incluir no system prompt do Claude


# ──────────────────────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────────────────────


def gerar_embedding(texto: str) -> list[float]:
    """Gera embedding da pergunta usando o mesmo modelo dos fundos."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def expandir_query(pergunta: str) -> str:
    """
    Reescreve a pergunta do usuário para incluir termos técnicos
    que existem nos documentos dos fundos, melhorando o retrieval.

    Ex: "isento de IR" → inclui "debêntures incentivadas infraestrutura tributação isento"
    """
    EXPANSION_PROMPT = """Você é um assistente que reescreve perguntas de clientes sobre investimentos
para melhorar a busca semântica em um banco de dados de fundos de investimento.

O banco contém fundos com campos como: nome, gestor, categoria, subcategoria, tipo_produto,
indexador, benchmark, tributacao, come_cotas, quando_indicar, quando_nao_indicar,
vantagens, desvantagens, alertas, descricao_tecnica.

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
    top_k: int = 5,
) -> list[dict]:
    """Chama a função RPC buscar_fundos no Supabase."""
    params = {
        "query_embedding": query_embedding,
        "top_k": top_k,
    }

    # Adiciona filtros apenas se fornecidos
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
    Busca dados complementares que a função buscar_fundos não retorna:
    - Plataformas disponíveis (fundo_plataformas)
    - Rentabilidade e PL (fundo_infos_atualizadas)
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

    # 2. Buscar dados atualizados (rentabilidade, PL)
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
        for campo in ["retorno_12m", "retorno_24m", "retorno_36m", "retorno_inicio", "pl_atual"]:
            if infos.get(campo) is not None:
                fundo[campo] = infos[campo]

        extras = extras_por_cnpj.get(cnpj, {})
        for campo in ["taxa_adm", "taxa_performance", "publico_alvo"]:
            if extras.get(campo) is not None:
                fundo[campo] = extras[campo]

    return fundos


def formatar_fundo_contexto(fundo: dict) -> FundoContexto:
    """Transforma o resultado da busca em contexto formatado para o prompt do Claude."""

    linhas = []
    linhas.append(f"📌 {fundo.get('nome', 'N/A')} (CNPJ: {fundo.get('cnpj', 'N/A')})")

    # --- Dados estruturados (da função buscar_fundos) ---
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

    # Taxas (campos extras enriquecidos)
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

    # --- Campos qualitativos (gerados pela IA nos pipelines) ---
    if fundo.get("quando_indicar"):
        linhas.append(f"   Quando indicar: {fundo['quando_indicar']}")
    if fundo.get("quando_nao_indicar"):
        linhas.append(f"   Quando NÃO indicar: {fundo['quando_nao_indicar']}")
    if fundo.get("vantagens"):
        linhas.append(f"   Vantagens: {fundo['vantagens']}")
    if fundo.get("desvantagens"):
        linhas.append(f"   Desvantagens: {fundo['desvantagens']}")
    if fundo.get("alertas"):
        linhas.append(f"   ⚠️ Alertas: {fundo['alertas']}")
    if fundo.get("descricao_tecnica"):
        linhas.append(f"   Descrição técnica: {fundo['descricao_tecnica']}")

    # --- Dados enriquecidos (fundo_infos_atualizadas) ---
    rents = []
    for periodo, label in [("retorno_12m", "12m"), ("retorno_24m", "24m"), ("retorno_36m", "36m")]:
        val = fundo.get(periodo)
        if val is not None:
            rents.append(f"{label}: {val:.2f}%")
    if rents:
        linhas.append(f"   Rentabilidade: {' | '.join(rents)}")
    if fundo.get("pl_atual") is not None:
        linhas.append(f"   PL atual: R$ {fundo['pl_atual']:,.0f}")

    # --- Plataformas (fundo_plataformas) ---
    if fundo.get("plataformas"):
        linhas.append(f"   Disponível em: {', '.join(fundo['plataformas'])}")

    contexto_texto = "\n".join(linhas)
    similaridade = fundo.get("similaridade", 0)

    return FundoContexto(
        cnpj=fundo.get("cnpj", ""),
        nome=fundo.get("nome", "N/A"),
        similaridade=round(similaridade, 4),
        contexto_texto=contexto_texto,
        dados_estruturados={
            k: v
            for k, v in fundo.items()
            if k not in ("embedding", "documento_texto") and v is not None
        },
    )


def montar_contexto_consolidado(
    fundos_formatados: list[FundoContexto],
    pergunta: str,
    filtros: dict,
) -> str:
    """Monta o bloco de contexto completo para incluir no prompt do Claude."""

    header = "=" * 60
    linhas = [
        header,
        "CONTEXTO: FUNDOS DE INVESTIMENTO RELEVANTES",
        header,
        f"Pergunta do cliente: {pergunta}",
    ]

    filtros_ativos = {k: v for k, v in filtros.items() if v}
    if filtros_ativos:
        linhas.append(f"Filtros aplicados: {json.dumps(filtros_ativos, ensure_ascii=False)}")

    linhas.append(f"Total de fundos encontrados: {len(fundos_formatados)}")
    linhas.append(header)

    for i, fundo in enumerate(fundos_formatados, 1):
        linhas.append(f"\n--- Fundo {i}/{len(fundos_formatados)} (similaridade: {fundo.similaridade:.4f}) ---")
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
    """Verificação de saúde da API."""
    return {
        "status": "ok",
        "service": "financeia-rag-api",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
    }


@app.post("/buscar", response_model=BuscaResponse)
async def buscar_fundos(
    request: BuscaRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Endpoint principal: recebe pergunta, busca fundos relevantes,
    retorna contexto formatado para o Base44 montar o prompt do Claude.

    Fluxo:
    1. Gera embedding da pergunta
    2. Chama buscar_fundos no Supabase (busca vetorial + filtros)
    3. Formata cada fundo como contexto legível
    4. Retorna contexto consolidado pronto para prompt
    """
    logger.info(f"🔍 Busca: '{request.pergunta}' | plataforma={request.plataforma} | plataformas={request.plataformas} | categoria={request.categoria}")

    try:
        # 1. Expandir a query para melhorar o retrieval
        query_expandida = expandir_query(request.pergunta)

        # 2. Gerar embedding da query expandida
        embedding = gerar_embedding(query_expandida)
        logger.info(f"   Embedding gerado ({len(embedding)} dimensões)")

    except Exception as e:
        logger.error(f"   ❌ Erro ao gerar embedding: {e}")
        raise HTTPException(status_code=502, detail=f"Erro ao gerar embedding: {str(e)}")

    try:
        # 3. Buscar fundos — se múltiplas plataformas, faz uma busca por plataforma e merge
        plataformas_lista = request.plataformas or ([request.plataforma] if request.plataforma else [None])

        todos_resultados = {}
        for plat in plataformas_lista:
            resultados_plat = buscar_fundos_supabase(
                query_embedding=embedding,
                plataforma=plat,
                categoria=request.categoria,
                tipo_produto=request.tipo_produto,
                top_k=request.top_k,
            )
            for r in resultados_plat:
                cnpj = r.get("cnpj")
                if cnpj not in todos_resultados:
                    todos_resultados[cnpj] = r

        # Reordenar por similaridade e limitar ao top_k
        resultados = sorted(
            todos_resultados.values(),
            key=lambda x: x.get("similaridade", 0),
            reverse=True,
        )[:request.top_k]

        logger.info(f"   {len(resultados)} fundos retornados pelo Supabase")

    except Exception as e:
        logger.error(f"   ❌ Erro na busca Supabase: {e}")
        raise HTTPException(status_code=502, detail=f"Erro na busca vetorial: {str(e)}")

    # 3. Enriquecer com dados complementares (plataformas, rentabilidade, taxas)
    resultados = enriquecer_com_dados_complementares(resultados)

    # 4. Formatar cada fundo
    fundos_formatados = [formatar_fundo_contexto(f) for f in resultados]

    # 5. Montar contexto consolidado
    filtros = {
        "plataforma": request.plataforma,
        "categoria": request.categoria,
        "tipo_produto": request.tipo_produto,
    }

    contexto = montar_contexto_consolidado(fundos_formatados, request.pergunta, filtros)

    return BuscaResponse(
        pergunta_original=request.pergunta,
        filtros_aplicados=filtros,
        total_resultados=len(fundos_formatados),
        fundos=fundos_formatados,
        contexto_consolidado=contexto,
    )


@app.post("/contexto-completo")
async def contexto_completo(
    request: BuscaRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Endpoint simplificado: retorna APENAS o contexto consolidado como texto.
    Útil se o Base44 só precisa do texto para colar no prompt do Claude,
    sem processar os dados estruturados.
    """
    resultado = await buscar_fundos(request, _api_key)
    return {"contexto": resultado.contexto_consolidado}


@app.get("/stats")
async def stats(_api_key: str = Depends(verify_api_key)):
    """Estatísticas básicas do catálogo."""
    try:
        # Total de fundos
        total = supabase_client.table("fundos").select("cnpj", count="exact").execute()

        # Total de embeddings
        embeddings = supabase_client.table("fundo_embeddings").select("cnpj", count="exact").execute()

        # Plataformas disponíveis
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
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/premissas", response_model=PremissasResponse)
async def listar_premissas(
    categoria: Optional[str] = None,
    _api_key: str = Depends(verify_api_key),
):
    """
    Retorna todas as premissas ativas, formatadas para incluir no system prompt.

    Uso pelo Base44:
    1. Chamar GET /premissas ao iniciar o chat (ou cachear por sessão)
    2. Incluir o campo 'contexto_premissas' no system prompt do Claude
    3. As premissas orientam TODAS as respostas do Claude

    Filtro opcional por categoria: macro, planejamento, alocacao, alertas, regras
    """
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
    """
    Atualiza uma premissa existente.
    Útil para o Base44 ter uma tela admin onde você edita premissas.

    Campos aceitos: titulo, categoria, conteudo, ativo
    """
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
    """
    Formata todas as premissas ativas em um bloco de texto
    pronto para incluir no system prompt do Claude.
    """
    if not premissas:
        return ""

    linhas = [
        "=" * 60,
        "PREMISSAS E DIRETRIZES DO ASSESSOR",
        "Considere SEMPRE as premissas abaixo ao elaborar suas respostas.",
        "=" * 60,
    ]

    # Agrupa por categoria para organização
    por_categoria = {}
    for p in premissas:
        cat = p.categoria or "geral"
        por_categoria.setdefault(cat, []).append(p)

    # Ordem de exibição das categorias
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