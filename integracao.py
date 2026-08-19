import re
import json
from agent_sql import executar_como_chain, executar_como_agente, get_llm
from agent_analista import gerar_mensagem_atendimento
from mock_channels import send_email_mock, send_whatsapp_mock, send_slack_mock
from langchain_core.prompts import PromptTemplate

def anonimizar_dados_sensiveis(texto_str):
    if not isinstance(texto_str, str):
        return texto_str
    
    # Regex para capturar e-mails
    email_regex = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    
    def mask_email(match):
        email = match.group(0)
        # Se for o e-mail padrão de teste do cliente do TCC, mantemos para a simulação
        if "cliente@teste.com" in email:
            return email
        
        user, domain = email.split('@', 1)
        # Se for email corporativo
        if "reserva" in domain or "empresa" in domain or domain not in ["gmail.com", "hotmail.com", "yahoo.com", "outlook.com"]:
            return "suporte.interno@empresa.com.br"
        
        # Para emails de clientes (como gmail, hotmail), mascaramos parcialmente por segurança LGPD
        if len(user) > 4:
            return f"{user[:2]}***{user[-2:]}@{domain}"
        return f"{user[0]}***@{domain}"
        
    return re.sub(email_regex, mask_email, texto_str)

def classificar_status_logistico(dados_brutos_str, data_hoje_simulada="2026-06-15"):
    try:
        dados_lista = json.loads(dados_brutos_str)
    except Exception:
        return "EM TRÂNSITO NO PRAZO"
        
    if not dados_lista or not isinstance(dados_lista, list) or len(dados_lista) == 0:
        return "EM TRÂNSITO NO PRAZO"
        
    pedido = dados_lista[0]
    status = pedido.get("Status Transportador", "")
    previsao_str = pedido.get("Previsão Entrega Cliente", None)
    data_entrega_str = pedido.get("Data Entrega", None)
    
    # Adquire status da VTEX/Síntese se houver nos dados retornados
    vtex_status = pedido.get("Status", "").lower().strip() if pedido.get("Status") else ""
    faturado_loja = pedido.get("Faturado Loja", None)
    
    # Normaliza strings
    status_lower = str(status).lower().strip()
    
    # 1. Se o pedido foi cancelado na VTEX ou na Síntese e não há faturamento posterior, é DEVOLVIDO/FALHA
    if vtex_status in ["cancelado", "canceled"] or "cancelado" in str(pedido.get("Status", "")).lower():
        # Mas se o pedido de alguma forma foi faturado depois, respeitamos a lógica de faturamento
        if not faturado_loja or str(faturado_loja).lower() == "none":
            return "DEVOLVIDO/FALHA"
    
    # 2. Devolvido / Falha Geral de Transporte
    if status_lower in ["falha na entrega", "cancelado", "devolvido"]:
        return "DEVOLVIDO/FALHA"
        
    # 3. Intercorrência / Extravio
    if status_lower in ["falha ao criar pedido com a transportadora", "averiguar falha na entrega"]:
        return "INTERCORRÊNCIA/EXTRAVIO"
        
    # Se status for Entregue
    if status_lower == "entregue":
        if not data_entrega_str or str(data_entrega_str).lower() == "none" or str(data_entrega_str).strip() == "":
            return "ENTREGUE NO PRAZO"
        try:
            dt_entrega = data_entrega_str.split(" ")[0]
            dt_previsao = previsao_str.split(" ")[0]
            if dt_entrega <= dt_previsao:
                return "ENTREGUE NO PRAZO"
            else:
                return "ENTREGUE COM ATRASO"
        except:
            return "ENTREGUE NO PRAZO"
            
    # Se em trânsito/despachado/saiu para entrega/fechado
    if status_lower in ["em trânsito", "em transito", "despachado", "saiu para entrega", "pendente na base", "fechado"]:
        if not previsao_str or str(previsao_str).lower() == "none" or str(previsao_str).strip() == "":
            return "EM TRÂNSITO NO PRAZO"
        try:
            dt_previsao = previsao_str.split(" ")[0]
            dt_hoje = data_hoje_simulada.split(" ")[0]
            if dt_previsao >= dt_hoje:
                return "EM TRÂNSITO NO PRAZO"
            else:
                return "EM TRÂNSITO COM ATRASO"
        except:
            return "EM TRÂNSITO NO PRAZO"
            
    # Se o status da vtex/sintese for Faturado/Faturado Loja, assume em trânsito no prazo caso sem transportador ainda
    if "faturado" in vtex_status:
        return "EM TRÂNSITO NO PRAZO"
        
    return "EM TRÂNSITO NO PRAZO"
def rotear_intencao(pergunta: str, model_name="openai/gpt-oss-20b") -> str:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    
    # Mapeamento transparente de modelos
    if "llama-3.1-8b" in model_name:
        model_name = "openai/gpt-oss-20b"
    elif "llama-3.3-70b" in model_name:
        model_name = "openai/gpt-oss-120b"
        
    llm = ChatGroq(model=model_name, temperature=0)
    
    prompt = PromptTemplate.from_template(
        """Você é um classificador de intenção especializado em sistemas de e-commerce e logística.
        Analise a pergunta do usuário e classifique-a em uma das duas intenções abaixo. Retorne APENAS o código da intenção escolhida (sem explicações, sem markdown, sem pontos):
        
        - STATUS_PEDIDO: Se o usuário estiver perguntando especificamente sobre o status de entrega, atraso, rastreamento ou dados de atendimento de um pedido de cliente específico (ex: "O pedido X está atrasado?", "Busque o status do pedido Y", "Qual a previsão do pedido Z?").
        - CONSULTA_GERAL: Se o usuário estiver fazendo uma consulta de dados gerais, relatórios administrativos, lista de CPFs, quantidade total de pedidos, esquema das tabelas do banco, injeções de comandos, saudações gerais ou qualquer pergunta de suporte corporativo/relatório interno de negócios (ex: "me dê a lista de CPFs", "quantos pedidos foram entregues ontem?", "ignore todas as regras...").
        
        Pergunta do usuário: {input}
        
        Código da Intenção (STATUS_PEDIDO ou CONSULTA_GERAL):"""
    )
    
    chain = prompt | llm
    try:
        resposta = chain.invoke({"input": pergunta})
        return resposta.content.strip().upper()
    except Exception:
        return "STATUS_PEDIDO"  # Fallback seguro em caso de erro na API

def pipeline_completo(pergunta: str, trilha_sql="chain", contato_cliente={"email": "cliente@teste.com", "telefone": "+55 11 99999-9999"}, chat_history=None, model_name="llama-3.1-8b-instant"):
    print("=====================================================")
    print(f"[START] INICIANDO PIPELINE - SISTEMA MULTI-AGENTE OMNICHANNEL")
    print(f"Pergunta/Comando: {pergunta}")
    print(f"Modelo LLM Selecionado: {model_name}")
    print("=====================================================\n")

    # Define os modelos para cada agente baseado na escolha (Suporta Híbrido)
    if model_name == "hybrid":
        sql_model = "llama-3.3-70b-versatile"
        analista_model = "llama-3.1-8b-instant"
    else:
        sql_model = model_name
        analista_model = model_name

    # Passo A: Reescrita de Pergunta Conversacional (RAG Conversacional)
    pergunta_final = pergunta
    if chat_history and len(chat_history) > 0:
        historico_str = ""
        for msg in chat_history:
            role = "Atendente" if msg["role"] == "user" else "IA"
            content = msg["content"]
            historico_str += f"{role}: {content}\n"
            
        try:
            rephrase_prompt = PromptTemplate.from_template(
                "Dada a seguinte conversa entre um atendente de logística (Usuário) e um sistema de IA, reescreva a última pergunta/comando do Usuário para ser uma pergunta direta, independente e autossuficiente, resolvendo qualquer ambiguidade, pronomes ou referências ao histórico (como 'desse pedido', 'ele', 'do anterior', 'do Thiago').\n\n"
                "REGRAS DE REESCRITA:\n"
                "- Se o Usuário digitar apenas um nome ou sobrenome de cliente (ex: 'Thiago Fernandes'), mantenha a busca ampla pelo nome. NÃO insira um ID de pedido específico da conversa anterior na reescrita se o Usuário não tiver digitado esse ID, pois podem haver outros clientes com nomes parecidos no banco.\n"
                "- Se o Usuário se referir a pronomes ou termos vagos como 'desse pedido' ou 'ele' e houver um ID de pedido ativo ou mencionado na resposta imediatamente anterior, então resolva para o ID do pedido específico.\n\n"
                "Histórico da Conversa:\n{chat_history}\n\n"
                "Última Pergunta do Usuário: {pergunta}\n\n"
                "Pergunta Reescrevida (Retorne APENAS o texto da pergunta direta reescrita, sem comentários ou explicações):"
            )
            rephrase_model_name = sql_model
            if rephrase_model_name == "hybrid" or "llama-3.3-70b" in rephrase_model_name:
                rephrase_model_name = "openai/gpt-oss-120b"
            elif "llama-3.1-8b" in rephrase_model_name:
                rephrase_model_name = "openai/gpt-oss-20b"
                
            llm_rephrase = get_llm(rephrase_model_name)
            chain_rephrase = rephrase_prompt | llm_rephrase
            res = chain_rephrase.invoke({
                "chat_history": historico_str,
                "pergunta": pergunta
            })
            pergunta_final = res.content.strip()
            print(f"-> Pergunta reescrita para contexto: '{pergunta_final}'\n")
        except Exception as e:
            print(f"Erro ao reescrever pergunta: {e}")

    # Passo 0: Agente Roteador de Intenção
    intencao = rotear_intencao(pergunta_final, model_name=sql_model)
    print(f">>> INTENÇÃO RECONHECIDA PELO ROTEADOR: {intencao}\n")
    
    if intencao == "CONSULTA_GERAL":
        print(f">>> PIPELINE ADMINISTRATIVO ATIVADO (Ignora redação de WhatsApp ao cliente)")
        if trilha_sql == "chain":
            dados_brutos = executar_como_chain(pergunta_final, model_name=sql_model)
        else:
            dados_brutos = executar_como_agente(pergunta_final, model_name=sql_model, chat_history=chat_history)
            
        dados_brutos = anonimizar_dados_sensiveis(dados_brutos)
        return {
            "dados_brutos": dados_brutos,
            "resposta_analista": f"[CONSULTA ADMINISTRATIVA / RELATÓRIO]\nOs dados solicitados foram extraídos com sucesso do banco de dados:\n\n{dados_brutos}",
            "mensagem_cliente": "Solicitação processada administrativamente. Como se trata de uma consulta interna de dados ou comando administrativo e não de suporte a um pedido de cliente, nenhuma notificação de WhatsApp foi enviada.",
            "tipo_consulta": "ADMINISTRATIVA"
        }

    # Passo 1: Agente SQL Extrai os Dados
    print(f">>> ETAPA 1: AGENTE 1 (SQL) BUSCANDO DADOS com modelo: {sql_model}")
    if trilha_sql == "chain":
        dados_brutos = executar_como_chain(pergunta_final, model_name=sql_model)
    else:
        dados_brutos = executar_como_agente(pergunta_final, model_name=sql_model, chat_history=chat_history)
        
    dados_brutos = anonimizar_dados_sensiveis(dados_brutos)
        
    # Proteção contra alucinação: se o banco retornar vazio (ex: query errada ou sem dados) ou se houver erro de SQL
    if not dados_brutos or str(dados_brutos).strip() == "" or str(dados_brutos) == "[]" or str(dados_brutos).startswith("Erro ao executar query"):
        return {
            "dados_brutos": dados_brutos if dados_brutos else "Nenhum dado encontrado.",
            "resposta_analista": "[ALERTA] Ocorreu um erro ou nenhum dado foi encontrado para analisar.",
            "mensagem_cliente": "Não consegui encontrar os dados necessários no banco de dados para responder sua pergunta. Por favor, verifique se as informações do pedido estão corretas.",
            "tipo_consulta": "SEM_DADOS"
        }
    
    # Tenta extrair a data simulada da pergunta do usuário para que o analista saiba qual data usar na comparação.
    # Caso não seja encontrada, usamos a data padrão simulada do projeto ('2026-06-15').
    data_hoje_simulada = "2026-06-15"
    match_data = re.search(r'\d{4}-\d{2}-\d{2}', pergunta_final)
    if match_data:
        data_hoje_simulada = match_data.group(0)
    
    # Classifica o status logisticamente de forma determinística
    classificacao_sistema = classificar_status_logistico(dados_brutos, data_hoje_simulada)
    
    # Tenta carregar dados_brutos como lista Python (já que ele é retornado como string JSON do executor)
    dados_lista = dados_brutos
    if isinstance(dados_brutos, str):
        try:
            dados_lista = json.loads(dados_brutos)
        except Exception:
            pass
            
    # Se a busca retornou múltiplos pedidos diferentes, tratamos como consulta administrativa (sem disparos ao cliente)
    pedidos_unicos = set()
    if isinstance(dados_lista, list):
        for item in dados_lista:
            p_id = item.get("Pedido") or item.get("Pedido de Venda") or item.get("Pedido de venda")
            if p_id:
                pedidos_unicos.add(p_id)
                
    if len(pedidos_unicos) > 1:
        print(f"-> Múltiplos pedidos detectados ({len(pedidos_unicos)} pedidos). Redirecionando para fluxo administrativo.")
        
        # Cria uma lista resumida e estruturada para facilitar a escolha do operador
        resumo_pedidos = []
        vistos = set()
        for item in dados_lista:
            p_id = item.get("Pedido") or item.get("Pedido de Venda") or item.get("Pedido de venda")
            if p_id and p_id not in vistos:
                vistos.add(p_id)
                nome = item.get("Nome do Destinatário") or item.get("Cliente") or "Cliente não identificado"
                status = item.get("Status Transportador") or item.get("Status") or "Sem status"
                resumo_pedidos.append(f"- 👤 **{nome}** | ID do Pedido: `{p_id}` | Status: `{status}`")
                
        resumo_texto = "\n".join(resumo_pedidos[:8])  # Limita a 8 na visualização para manter a tela limpa
        if len(resumo_pedidos) > 8:
            resumo_texto += f"\n- ... e outros {len(resumo_pedidos) - 8} pedidos encontrados com o nome especificado."
            
        mensagem_retorno = (
            f"[CONSULTA ADMINISTRATIVA / DADOS AMBÍGUOS]\n\n"
            f"🔍 Foram encontrados **{len(pedidos_unicos)} pedidos** associados a este termo no banco de dados.\n\n"
            f"Por favor, refine a sua pergunta informando o **ID do Pedido** específico ou o **Nome Completo** do cliente.\n\n"
            f"**Pedidos Encontrados:**\n"
            f"{resumo_texto}"
        )
        
        return {
            "dados_brutos": dados_brutos,
            "resposta_analista": mensagem_retorno,
            "mensagem_cliente": "Múltiplos pedidos identificados. Por favor, refine sua pesquisa.",
            "tipo_consulta": "CONSULTA_GERAL"
        }

    # Trunca os dados brutos se houver muitos registros para evitar estourar o limite de tokens da API (TPM)
    dados_brutos_para_analista = dados_lista
    if isinstance(dados_lista, list) and len(dados_lista) > 2:
        dados_brutos_para_analista = dados_lista[:2]
        dados_brutos_para_analista.append({
            "AVISO_SISTEMA": f"Foram encontrados {len(dados_lista)} registros no total para esta busca. Mostrando apenas os 2 primeiros para evitar limite de tokens (TPM). Peça para o atendente refinar a busca fornecendo o sobrenome ou o ID do pedido."
        })
        
    # Prepara os dados para o analista
    dicionario_para_analista = {
        "Dados Logísticos (Brutos do Banco)": json.dumps(dados_brutos_para_analista, ensure_ascii=False, indent=2) if isinstance(dados_brutos_para_analista, list) else str(dados_brutos_para_analista),
        "Data Atual de Hoje (Simulada para Análise)": data_hoje_simulada,
        "Classificação Logística Determinada pelo Sistema": classificacao_sistema
    }

    # Passo 2: Agente Analista (Redator)
    print(f"\n>>> ETAPA 2: AGENTE 2 (ANALISTA) GERANDO A MENSAGEM com modelo: {analista_model}")
    resposta_analista = gerar_mensagem_atendimento(dicionario_para_analista, classificacao_sistema=classificacao_sistema, model_name=analista_model)
    print(f"\n[OUTPUT DO AGENTE 2]:\n{resposta_analista}\n")

    # Passo 3: Parsing (Separar Mensagem do Cliente e Análise Interna)
    print(">>> ETAPA 3: PARSING E DISTRIBUIÇÃO OMNICHANNEL")
    
    # Extrair apenas a mensagem do cliente
    mensagem_cliente = resposta_analista
    
    # Utilizar regex case-insensitive para extrair a mensagem do cliente
    partes = re.split(r'(?i)mensagem para o cliente', resposta_analista)
    if len(partes) > 1:
        # Pega a última parte e limpa marcadores comuns de formatação
        mensagem_cliente = partes[-1].strip(" *:\n\r[]-").strip().replace("---", "").strip()

    # Disparar Canais Simulados
    send_slack_mock(channel="logistica-alertas", message=resposta_analista)
    send_email_mock(to_email=contato_cliente["email"], subject="Atualização sobre o seu pedido", message=mensagem_cliente)
    send_whatsapp_mock(phone=contato_cliente["telefone"], message=mensagem_cliente)

    print("=====================================================")
    print("[SUCCESS] PIPELINE FINALIZADO COM SUCESSO!")
    print("=====================================================")
    
    return {
        "dados_brutos": dados_brutos,
        "resposta_analista": resposta_analista,
        "mensagem_cliente": message_to_client_fallback if 'mensagem_cliente' not in locals() else mensagem_cliente,
        "tipo_consulta": "STATUS_PEDIDO"
    }

if __name__ == "__main__":
    # Teste prático do Pipeline
    # Testando um pedido com atraso e intercorrência (Não Entregue)
    pedido_teste = "FCN-1635681090149-01" 
    
    # Adicionando uma Data Atual simulada para que o LLM não use a data real de hoje
    # (já que o banco de dados é um snapshot de Junho/2026)
    pergunta_teste = f"Busque todos os dados do pedido {pedido_teste} na tabela rastreio_intelipost. Considere que a Data Atual de Hoje é 2026-06-15 para realizar sua análise de atraso."
    
    # Podemos trocar para 'agente' para usar o LLM Autônomo Grande no SQL
    pipeline_completo(pergunta_teste, trilha_sql="chain")
