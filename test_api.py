"""
Script de teste local da API.
Roda com: python test_api.py

Requer:
- API rodando localmente (python main.py)
- Variáveis de ambiente configuradas
"""

import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "financeia-rag-2026-prod"  # Mesma do .env

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


def test_health():
    print("=" * 50)
    print("1. Health Check")
    print("=" * 50)
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    print()


def test_stats():
    print("=" * 50)
    print("2. Stats")
    print("=" * 50)
    r = requests.get(f"{BASE_URL}/stats", headers=HEADERS)
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    print()


def test_busca_simples():
    print("=" * 50)
    print("3. Busca simples (sem filtros)")
    print("=" * 50)
    payload = {
        "pergunta": "Qual fundo de renda fixa com boa liquidez para perfil conservador?",
        "top_k": 3,
    }
    r = requests.post(f"{BASE_URL}/buscar", headers=HEADERS, json=payload)
    print(f"Status: {r.status_code}")

    data = r.json()
    print(f"Total resultados: {data['total_resultados']}")
    for f in data["fundos"]:
        print(f"  - {f['nome']} (sim: {f['similaridade']:.4f})")
    print()
    print("Contexto consolidado (primeiras 500 chars):")
    print(data["contexto_consolidado"][:500])
    print()


def test_busca_com_filtro():
    print("=" * 50)
    print("4. Busca com filtro de plataforma")
    print("=" * 50)
    payload = {
        "pergunta": "Fundo multimercado com proteção contra inflação",
        "plataforma": "XP",
        "top_k": 3,
    }
    r = requests.post(f"{BASE_URL}/buscar", headers=HEADERS, json=payload)
    print(f"Status: {r.status_code}")

    data = r.json()
    print(f"Total resultados: {data['total_resultados']}")
    print(f"Filtros: {json.dumps(data['filtros_aplicados'], ensure_ascii=False)}")
    for f in data["fundos"]:
        print(f"  - {f['nome']} (sim: {f['similaridade']:.4f})")
    print()


def test_contexto_completo():
    print("=" * 50)
    print("5. Endpoint /contexto-completo")
    print("=" * 50)
    payload = {
        "pergunta": "Previdência privada para longo prazo com isenção de come-cotas",
        "top_k": 3,
    }
    r = requests.post(f"{BASE_URL}/contexto-completo", headers=HEADERS, json=payload)
    print(f"Status: {r.status_code}")

    data = r.json()
    print("Contexto (primeiras 800 chars):")
    print(data["contexto"][:800])
    print()


def test_auth_invalida():
    print("=" * 50)
    print("6. Teste de autenticação inválida")
    print("=" * 50)
    r = requests.post(
        f"{BASE_URL}/buscar",
        headers={"Content-Type": "application/json"},
        json={"pergunta": "teste"},
    )
    print(f"Status: {r.status_code} (esperado: 401)")
    print(json.dumps(r.json(), indent=2))
    print()


def test_premissas():
    print("=" * 50)
    print("7. Listar premissas ativas")
    print("=" * 50)
    r = requests.get(f"{BASE_URL}/premissas", headers=HEADERS)
    print(f"Status: {r.status_code}")

    data = r.json()
    print(f"Total premissas: {data['total']}")
    for p in data["premissas"]:
        print(f"  - [{p.get('categoria', 'geral')}] {p['titulo']}")
    print()
    if data["contexto_premissas"]:
        print("Contexto formatado (primeiras 500 chars):")
        print(data["contexto_premissas"][:500])
    print()


def test_premissas_por_categoria():
    print("=" * 50)
    print("8. Premissas filtradas por categoria")
    print("=" * 50)
    r = requests.get(f"{BASE_URL}/premissas?categoria=macro", headers=HEADERS)
    print(f"Status: {r.status_code}")

    data = r.json()
    print(f"Total premissas (macro): {data['total']}")
    for p in data["premissas"]:
        print(f"  - {p['titulo']}")
    print()


if __name__ == "__main__":
    print("\n🧪 Testando FinanceIA RAG API\n")

    test_health()
    test_stats()
    test_busca_simples()
    test_busca_com_filtro()
    test_contexto_completo()
    test_auth_invalida()
    test_premissas()
    test_premissas_por_categoria()

    print("✅ Todos os testes executados!")
