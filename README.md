# FinanceIA RAG API

API intermediária que conecta o Base44 ao catálogo de fundos e premissas no Supabase/pgvector.

## Arquitetura

```
Usuário faz pergunta no FinanceIA (Base44)
        ↓
Base44 coleta dados do usuário (perfil, patrimônio, objetivos)
        ↓
Base44 chama API:
  ├── GET  /premissas  → premissas e diretrizes ativas
  └── POST /buscar     → fundos relevantes via busca vetorial
        ↓
Base44 monta o prompt:
  [System]  Premissas + Regras do assessor
  [Context] Fundos encontrados + Dados do cliente
  [User]    Pergunta original
        ↓
Base44 chama Claude API → Resposta ao usuário
```

## Fontes de Conhecimento

| Fonte | Onde vive | Como entra no prompt |
|-------|-----------|---------------------|
| Dados do usuário | Base44 | Base44 acessa direto |
| Catálogo de fundos | Supabase (busca vetorial) | API `/buscar` retorna top-k relevantes |
| Premissas/guidance | Supabase (tabela simples) | API `/premissas` retorna todas ativas |

## Setup Local

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar variáveis de ambiente
cp .env.example .env
# Editar .env com suas credenciais

# 3. Criar tabela de premissas no Supabase
# Rodar sql_premissas.sql no SQL Editor do Supabase

# 4. Rodar
python main.py
# API disponível em http://localhost:8000

# 5. Testar
python test_api.py

# 6. Documentação interativa
# Abrir http://localhost:8000/docs no navegador
```

## Deploy

### Railway
```bash
npm install -g @railway/cli
railway login
railway init
railway up
# Configurar variáveis no painel do Railway
```

### Render
- Criar Web Service no dashboard do Render
- Conectar repositório Git
- O render.yaml configura tudo automaticamente
- Adicionar variáveis de ambiente no painel

## Endpoints

### `GET /health`
Verificação de saúde (sem autenticação).

### `POST /buscar`
Busca semântica de fundos. Requer header `X-API-Key`.

```json
{
  "pergunta": "Fundo de renda fixa para perfil conservador",
  "plataforma": "XP",
  "categoria": "Renda Fixa",
  "tipo_produto": "Fundos",
  "top_k": 5
}
```

Retorna fundos com contexto formatado + dados estruturados.

### `POST /contexto-completo`
Versão simplificada do `/buscar` — retorna apenas o texto de contexto.

### `GET /premissas`
Retorna todas as premissas ativas, formatadas para o system prompt.
Filtro opcional: `?categoria=macro`

### `GET /stats`
Estatísticas do catálogo (total de fundos, embeddings, plataformas).

### `PUT /premissas/{id}`
Atualiza uma premissa (titulo, categoria, conteudo, ativo).

## Integração com Base44

```javascript
const API_URL = "https://sua-api.railway.app";
const API_KEY = "sua-api-key";
const headers = { "Content-Type": "application/json", "X-API-Key": API_KEY };

// 1. Buscar premissas (cachear por sessão — não muda a cada pergunta)
const premissasResp = await fetch(`${API_URL}/premissas`, { headers });
const { contexto_premissas } = await premissasResp.json();

// 2. Buscar fundos relevantes para a pergunta
const fundosResp = await fetch(`${API_URL}/buscar`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    pergunta: perguntaDoUsuario,
    plataforma: perfilCliente.plataforma,
    top_k: 5
  })
});
const { contexto_consolidado } = await fundosResp.json();

// 3. Montar prompt com as 3 fontes de conhecimento
const systemPrompt = `
Você é o assessor de investimentos da 3A Riva Investimentos.

${contexto_premissas}
`;

const userPrompt = `
Dados do cliente:
- Perfil: ${cliente.perfil}
- Patrimônio: R$ ${cliente.patrimonio}
- Objetivo principal: ${cliente.objetivo}

${contexto_consolidado}

Pergunta: ${perguntaDoUsuario}
`;

// 4. Chamar Claude API
const claudeResponse = await callClaudeAPI(systemPrompt, userPrompt);
```

## Arquivos

| Arquivo | Função |
|---------|--------|
| `main.py` | API FastAPI (endpoints, busca vetorial, formatação) |
| `requirements.txt` | Dependências Python |
| `.env.example` | Template de variáveis de ambiente |
| `sql_premissas.sql` | SQL para criar tabela + dados iniciais de exemplo |
| `test_api.py` | Script de testes locais (8 cenários) |
| `Dockerfile` | Container para deploy |
| `railway.toml` | Config Railway |
| `render.yaml` | Config Render |
