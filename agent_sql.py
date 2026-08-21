import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Carrega as chaves de API do arquivo .env
load_dotenv()

# ==========================================
# 1. CONEXÃO COM O BANCO DE DADOS
# ==========================================
db_path = r"c:\Users\gabri\OneDrive\Documentos\Pós - BI\TCC - pós\banco_tcc.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Variável global para armazenar a última query SQL gerada pela IA
LAST_GENERATED_SQL = "Nenhuma query gerada ainda."


def get_llm(model_name="openai/gpt-oss-20b"):
    from langchain_groq import ChatGroq
    # Mapeia dinamicamente modelos deprecados para novos ativos caso passem o nome antigo
    if "llama-3.1-8b" in model_name:
        model_name = "openai/gpt-oss-20b"
    elif "llama-3.3-70b" in model_name:
        model_name = "openai/gpt-oss-120b"
        
    return ChatGroq(model=model_name, temperature=0)

# ==========================================
# 3. TRILHA 1: AGENTE AUTÔNOMO (Para LLMs Grandes)
# ==========================================
def executar_como_agente(pergunta: str, model_name="openai/gpt-oss-20b", chat_history=None):
    if model_name == "hybrid" or "llama-3.3-70b" in model_name:
        model_name = "openai/gpt-oss-120b"
    elif "llama-3.1-8b" in model_name:
        model_name = "openai/gpt-oss-20b"
    llm = get_llm(model_name)
    
    # Injetando histórico para dar memória ao agente
    if chat_history and len(chat_history) > 0:
        # Pega apenas as últimas 2 interações para poupar Tokens (Evita erro 413 da Groq)
        chat_history = chat_history[-2:]
        contexto_hist = "=== HISTÓRICO DA CONVERSA (Contexto) ===\n"
        for msg in chat_history:
            role = "Usuário" if msg["role"] == "user" else "Agente"
            contexto_hist += f"{role}: {msg['content']}\n"
        contexto_hist += "========================================\n"
        contexto_hist += "USE O HISTÓRICO ACIMA PARA ENTENDER O CONTEXTO DA NOVA PERGUNTA, SE NECESSÁRIO (ex: pronomes como 'eles', 'esse pedido').\n"
        pergunta = f"{contexto_hist}\nNova Pergunta do Usuário: {pergunta}"
    
    dicionario_dados = """\nDICIONÁRIO DE DADOS E REGRAS DE NEGÓCIO:
- DATA ATUAL DE HOJE (Padrão): O banco de dados é um snapshot histórico de Junho de 2026. Se a pergunta não mencionar explicitamente uma data atual, ASSUMA SEMPRE que a data atual é '2026-06-15'.
- ID do Pedido: Na tabela 'sintese_pedidos' e 'sintese_recusas', busque na coluna "Pedido". Na tabela 'rastreio_intelipost', busque na coluna "Pedido de Venda".
- Status Real da Entrega e Rastreio: A fonte da verdade do rastreio é SEMPRE a tabela 'rastreio_intelipost' (NÃO use pedidos_vtex). Os valores de "Status Transportador" nesta tabela são: 'Entregue', 'Em trânsito', 'Despachado', 'Saiu para entrega', 'Falha na entrega', 'Averiguar falha na entrega', 'Fechado', 'Cancelado'.
- Pedidos em Trânsito: Um pedido está em trânsito se "Status Transportador" for diferente de 'Entregue' e diferente de 'Cancelado'.
- Atrasos: Um pedido está em atraso se a "Previsão Entrega Cliente" for menor que a data de hoje (ou a simulada, ex: '2026-06-15') E o status for diferente de 'Entregue' e diferente de 'Cancelado'. ATENÇÃO: Colunas com espaço DEVEM estar entre aspas duplas no SQLite (ex: "Previsão Entrega Cliente").
- Entregue com Atraso (Entregue Fora do Prazo): Um pedido foi entregue com atraso se o "Status Transportador" for igual a 'Entregue' E a "Data Entrega" for maior que a "Previsão Entrega Cliente" (ambas formatadas como YYYY-MM-DD).
- Entregue no Prazo: Um pedido foi entregue no prazo se o "Status Transportador" for igual a 'Entregue' E a "Data Entrega" for menor ou igual à "Previsão Entrega Cliente".
- Intercorrência: Um pedido tem alguma intercorrência na entrega se "Status Transportador" for 'Falha na entrega', 'Falha ao criar pedido com a transportadora' ou 'Averiguar falha na entrega'.
- SELEÇÃO DE COLUNAS: Ao buscar um pedido ou o último pedido para análise de atendimento, a query SQL deve obrigatoriamente trazer TODAS as colunas (`SELECT *` ou selecionar explicitamente colunas como "Pedido de Venda", "Status Transportador", "Descrição Transportador", "Previsão Entrega Cliente", "Pagina Rastreamento", "e-mail Destinatário", "Celular Destinatário") e NÃO apenas a coluna do ID do pedido. O Agente Analista precisa de todos esses dados para trabalhar.
- ORDENAÇÃO DE DATAS NO SQLITE (IMPORTANTE): As colunas "Data Hora Recusa" ou "Data" podem estar salvas como texto no formato brasileiro 'DD/MM/YYYY' ou 'DD/MM/YYYY HH:MM:SS' (ex: 30/06/2026). Para ordenar essas datas cronologicamente (ex: obter o último pedido recusado), o SQLite ordena incorretamente por string. Você DEVE ordenar convertendo a string de data para o formato cronológico 'YYYY-MM-DD' usando substring: `ORDER BY substr("Data Hora Recusa", 7, 4) || '-' || substr("Data Hora Recusa", 4, 2) || '-' || substr("Data Hora Recusa", 1, 2) DESC` (ou adaptando os índices da substring para o campo de data correspondente).
- BUSCA POR ID NA TABELA PEDIDOS_VTEX (IMPORTANTE): Os códigos de pedido na tabela `pedidos_vtex` estão salvos SEM o prefixo "FCN-" (ex: '1642991104922-01'). Ao buscar ou cruzar dados com a tabela `pedidos_vtex`, você DEVE usar o operador `LIKE` para ignorar o prefixo (ex: `WHERE "Order" LIKE '%1642991104922-01'`) ou utilizar `REPLACE("Pedido", 'FCN-', '')` para remover o prefixo.
- INTEGRAÇÃO DE RECUSAS NO STATUS (IMPORTANTE): Ao buscar os dados completos de status de um pedido específico (para entender o histórico dele de ponta a ponta), você DEVE sempre incluir informações da tabela `sintese_recusas` correspondentes àquele pedido se elas existirem (use por exemplo um `LEFT JOIN` da tabela principal com a `sintese_recusas` ou traga todos os registros relacionados em uma busca conjunta para que o analista saiba se o pedido passou por uma recusa antes de seguir o fluxo de faturamento e entrega).
- EVITE FILTROS CONDICIONAIS DINÂMICOS NO WHERE (CRÍTICO): Ao formular queries para responder a perguntas específicas sobre se um pedido foi entregue com atraso ou no prazo (ex: "ele foi entregue com atraso?"), filtre na cláusula WHERE apenas pelo identificador do pedido (ex: `WHERE "Pedido de Venda" = 'FCN-XXXX'`). NUNCA adicione filtros condicionais baseados na pergunta (como `AND "Data Entrega" > "Previsão Entrega Cliente"` ou `AND "Status Transportador" = 'Entregue'`) na cláusula WHERE. Se você adicionar estes filtros e a condição for falsa (ex: o pedido não atrasou), a query retornará vazia (0 linhas), impedindo o sistema de obter os dados do pedido. O banco deve retornar a linha do pedido, e a lógica de verificação de atraso/status será feita posteriormente pelo Agente Analista.
"""
    
    cot_prefix = f"""Você é um especialista em logística e banco de dados.{dicionario_dados}
Pense passo a passo:
1. Entenda a pergunta e decida qual tabela usar com base no Dicionário.
2. Formule uma query SQLite válida usando as colunas corretas.
3. Execute a query usando a ferramenta do banco de dados.
4. Responda ao usuário com o resultado.
"""
    
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        prefix=cot_prefix,
        agent_executor_kwargs={"handle_parsing_errors": True},
        verbose=False
    )
    
    import sys
    import io
    
    # Previne que a captura do terminal quebre por caracteres Unicode/UTF-8
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        resposta = agent_executor.invoke({"input": pergunta})
    finally:
        sys.stdout = old_stdout
        
    return resposta["output"]

# ==========================================
# 4. TRILHA 2: PIPELINE / CHAIN (Para SLMs Pequenos como Llama)
# ==========================================
def executar_como_chain(pergunta: str, model_name="openai/gpt-oss-20b"):
    if model_name == "hybrid" or "llama-3.3-70b" in model_name:
        model_name = "openai/gpt-oss-120b"
    elif "llama-3.1-8b" in model_name:
        model_name = "openai/gpt-oss-20b"
    print(f"\n[TRILHA CHAIN] Extraindo query SQL da pergunta com {model_name}...")
    llm = get_llm(model_name)
    
    # Passo A: O modelo gera APENAS a query SQL baseada no schema
    prompt_template = PromptTemplate.from_template(
        """Dada a pergunta do usuário, o Dicionário de Dados e o schema do banco de dados abaixo, escreva UMA query SQLite válida para responder à pergunta. 
        Retorne APENAS a query SQL em texto puro, sem a palavra "sql", sem formatação markdown ou crases.
        
        DICIONÁRIO DE DADOS E REGRAS:
        - DATA ATUAL DE HOJE (Padrão): O banco de dados é um snapshot de Junho de 2026. Se a pergunta não mencionar explicitamente uma data atual, ASSUMA SEMPRE que a data atual é '2026-06-15'.
        - ID do Pedido: Na tabela 'sintese_pedidos' e 'sintese_recusas', o ID do pedido fica na coluna "Pedido". Na tabela 'rastreio_intelipost', fica na coluna "Pedido de Venda".
        - Fonte da Verdade do Rastreio: É sempre a tabela 'rastreio_intelipost' (NÃO use pedidos_vtex para status de trânsito ou rastreio).
        - Pedidos em Trânsito: "Status Transportador" na tabela 'rastreio_intelipost' diferente de 'Entregue' e diferente de 'Cancelado'.
        - Pedidos Atrasados: "Previsão Entrega Cliente" < a data atual (ou a simulada '2026-06-15') E "Status Transportador" diferente de 'Entregue' e diferente de 'Cancelado'.
        - Entregue com Atraso (Entregue Fora do Prazo): Ocorre quando "Status Transportador" é igual a 'Entregue' E a data em "Data Entrega" é maior que "Previsão Entrega Cliente" (ambas formatadas como YYYY-MM-DD).
        - Entregue no Prazo: Ocorre quando "Status Transportador" é igual a 'Entregue' E a data em "Data Entrega" é menor ou igual à "Previsão Entrega Cliente".
        - Intercorrência: Ocorre quando "Status Transportador" for 'Falha na entrega', 'Falha ao criar pedido com a transportadora' ou 'Averiguar falha na entrega'.
        - SELEÇÃO DE COLUNAS: Ao buscar um pedido ou o último pedido para análise de atendimento, a query SQL deve obrigatoriamente trazer TODAS as colunas (`SELECT *` ou trazer "Pedido de Venda", "Status Transportador", "Descrição Transportador", "Previsão Entrega Cliente", "Pagina Rastreamento", "e-mail Destinatário", "Celular Destinatário") e NÃO apenas a coluna de ID. O redator precisa dessas informações.
        
        ATENÇÃO/REGRAS DE SQL:
        - BUSCA DE TEXTOS (LIKE): Sempre que filtrar por nomes de pessoas, clientes, cidades ou transportadoras na cláusula WHERE (ex: filtro de cliente), utilize o operador LIKE com curingas (ex: p."Cliente" LIKE '%Thiago%') em vez do operador de igualdade (=).
        - Colunas que possuem espaços no nome DEVEM ser envolvidas em aspas duplas na query (Ex: "Pedido de Venda").
        - ORDENAÇÃO DE DATAS NO SQLITE: As colunas "Data Hora Recusa" ou "Data" podem estar salvas como texto no formato 'DD/MM/YYYY'. Para ordenar cronologicamente de forma correta, use substring: `ORDER BY substr("Data Hora Recusa", 7, 4) || '-' || substr("Data Hora Recusa", 4, 2) || '-' || substr("Data Hora Recusa", 1, 2) DESC`.
        - BUSCA POR ID NA TABELA PEDIDOS_VTEX: Os códigos de pedido na tabela `pedidos_vtex` estão salvos SEM o prefixo "FCN-" (ex: '1642991104922-01'). Ao buscar ou cruzar dados com a tabela `pedidos_vtex`, use `LIKE` para ignorar o prefixo.
        - INTEGRAÇÃO DE RECUSAS NO STATUS: Ao buscar dados de status de um pedido específico, tente sempre incluir/trazer informações da tabela `sintese_recusas` correspondentes àquele pedido se elas existirem (ex: fazendo um `LEFT JOIN` ou incluindo em consultas juntas), para sabermos se o pedido sofreu recusa prévia de seller antes de ser faturado/entregue.
        - EVITE FILTROS CONDICIONAIS DINÂMICOS NO WHERE (CRÍTICO): Ao formular queries para responder a perguntas específicas sobre se um pedido foi entregue com atraso ou no prazo (ex: "ele foi entregue com atraso?"), filtre na cláusula WHERE apenas pelo identificador do pedido (ex: `WHERE "Pedido de Venda" = 'FCN-XXXX'`). NUNCA adicione filtros condicionais baseados na pergunta (como `AND "Data Entrega" > "Previsão Entrega Cliente"` ou `AND "Status Transportador" = 'Entregue'`) na cláusula WHERE. Se você adicionar estes filtros e a condição for falsa (ex: o pedido não atrasou), a query retornará vazia (0 linhas), impedindo o sistema de obter os dados do pedido. O banco deve retornar a linha do pedido, e a lógica de verificação de atraso/status será feita de forma determinística posteriormente pelo Agente Analista.
        
        Schema das tabelas:
        {table_info}
        
        Pergunta do Usuário: {input}
        
        Sua Query SQL (APENAS A QUERY E MAIS NADA):"""
    )
    
    # Busca o schema do banco de dados resumido para não estourar o limite de tokens da Groq
    table_info = """
    Tabela rastreio_intelipost: "Pedido de Venda" (string), "Pedido" (string), "Status Transportador" (string), "Data Entrega" (string), "Previsão Entrega Cliente" (string), "Pagina Rastreamento" (string), "e-mail Destinatário" (string), "Celular Destinatário" (string), "Valor Total" (string)
    Tabela sintese_pedidos: "Pedido" (string), "Data" (string), "Status" (string), "Cliente" (string), "CPF" (string)
    Tabela sintese_recusas: "Pedido" (string), "Data Hora Recusa" (string), "Motivo Recusa" (string), "Descrição Produto - Cor" (string)
    Tabela pedidos_vtex: "Order" (string), "Creation Date" (string), "Client Name" (string), "Email" (string), "Phone" (string), "Status" (string), "Estimate Delivery Date" (string)
    """
    
    # Cria a Chain (Pipeline) de forma pura (LCEL)
    chain_sql = prompt_template | llm
    
    # Executa a Chain
    resposta_llm = chain_sql.invoke({"input": pergunta, "table_info": table_info})
    query_gerada = resposta_llm.content
    
    # Limpeza básica caso o modelo ainda retorne crases markdown teimosamente
    query_limpa = query_gerada.replace("```sql", "").replace("```", "").strip()
    print(f"-> Query SQL Gerada pela IA:\n{query_limpa}\n")
    
    # Atualiza a variável global com a última query
    global LAST_GENERATED_SQL
    LAST_GENERATED_SQL = query_limpa
    
    # Passo B: O sistema (Python) executa a query no banco de dados com segurança
    print("-> Executando query no banco...")
    import sqlite3
    import json
    
    # Validação de segurança (Guardrail de Banco de Dados): Permite estritamente consultas de leitura (SELECT ou WITH)
    query_temp = query_limpa.strip().upper()
    if not (query_temp.startswith("SELECT") or query_temp.startswith("WITH")):
        print("-> [BLOQUEIO DE SEGURANÇA] Tentativa de modificação de dados ou query destrutiva detectada!")
        return "Erro de segurança: Apenas consultas de leitura (SELECT) são permitidas no banco de dados. Ações de modificação ou destruição foram bloqueadas."
        
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query_limpa)
        rows = cursor.fetchall()
        resultados = [dict(row) for row in rows]
        conn.close()
        
        if resultados:
            resultado_banco = json.dumps(resultados, ensure_ascii=False, indent=2)
        else:
            resultado_banco = "[]"
    except Exception as e:
        resultado_banco = f"Erro ao executar query: {str(e)}"
        
    # Para a Fase 2 do TCC, o objetivo final desta etapa é retornar a "View" dos dados
    return resultado_banco

if __name__ == "__main__":
    pergunta_teste = "Encontre o pedido 'FCN-1513080904883-01' na tabela rastreio_intelipost e me diga qual é o status_transportador."
    
    print("===== TESTE TRILHA CHAIN (LLAMA 3 / GROQ) =====")
    try:
        resultado_chain = executar_como_chain(pergunta_teste, model_name="llama-3.1-8b-instant")
        print(f"\n[View Final retornada pelo Banco de Dados]:\n{resultado_chain}")
    except Exception as e:
        print(f"Erro na Trilha Chain: {e}")

    print("\n===== TESTE TRILHA AGENT (LLAMA 3.3 70B) =====")
    try:
        resultado_agent = executar_como_agente(pergunta_teste, model_name="llama-3.3-70b-versatile")
        print(f"\n[Resposta Final do Agente]: {resultado_agent}")
    except Exception as e:
        print(f"Erro na Trilha Agent: {e}")
