import sys
import io

# Configura o terminal para aceitar UTF-8 no Windows, evitando erros de charmap codec ao printar/logar
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import streamlit as st
import time
import pandas as pd
import json
from agent_sql import executar_como_agente, executar_como_chain
from integracao import pipeline_completo

# Configuração da Página e Estilos CSS Modernos
st.set_page_config(page_title="TCC - Agente IA Logístico", page_icon="📦", layout="wide")

# Custom CSS para estética premium e cards de comunicação simulados
st.markdown("""
<style>
    .whatsapp-card {
        background-color: #dcf8c6;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #075e54;
        color: #303030 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .whatsapp-header {
        font-weight: bold;
        color: #075e54;
        margin-bottom: 5px;
        font-size: 0.9em;
    }
    .whatsapp-footer {
        text-align: right;
        font-size: 0.8em;
        color: #757575;
        margin-top: 5px;
    }
    .email-card {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        color: #222222 !important;
    }
    .email-subject {
        font-size: 1.25em;
        font-weight: bold;
        color: #111111;
        margin-bottom: 12px;
        border-bottom: 2px solid #db4437;
        padding-bottom: 8px;
    }
    .email-meta {
        font-size: 0.85em;
        color: #555555;
        background-color: #f9f9f9;
        padding: 8px 12px;
        border-radius: 6px;
        margin-bottom: 15px;
        line-height: 1.4;
    }
    .email-body {
        font-size: 0.95em;
        line-height: 1.6;
        color: #333333;
    }
    .email-signature {
        margin-top: 20px;
        padding-top: 12px;
        border-top: 1px dashed #e0e0e0;
        font-size: 0.85em;
        color: #777777;
        line-height: 1.4;
    }
    .slack-card {
        background-color: #f8f8f8;
        border: 1px solid #dddddd;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4a154b;
        color: #2c2c2c !important;
        font-family: 'Lato', sans-serif;
    }
    .slack-header {
        font-weight: bold;
        color: #4a154b;
        margin-bottom: 5px;
        font-size: 0.9em;
    }
    .title-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .title-banner h1 {
        color: white !important;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Banner de Título Superior
st.markdown("""
<div class="title-banner">
    <h1>🤖 Sistema Inteligente Multi-Agente de Logística</h1>
    <p style="margin-top: 8px; font-size: 1.1em; opacity: 0.9;">
        Plataforma Inteligente para Análise de Entregas, Guardrails Anti-Alucinação e Disparos Omnichannel.
    </p>
</div>
""", unsafe_allow_html=True)

# Menu lateral para escolher as configurações do TCC
st.sidebar.header("🛠️ Configurações da IA")

trilha = st.sidebar.radio(
    "Escolha a Arquitetura de Execução:",
    ("Sistema Multi-Agente Omnichannel", "Agente de Dados Autônomo (Text-to-SQL)", "Pipeline Direto (Apenas Extração SQL)")
)

# Seleção Dinâmica de LLMs
modelo_opcao = st.sidebar.selectbox(
    "Selecione o Modelo de Linguagem (Groq):",
    ("GPT-OSS 20B (Rápido e Eficiente)", "GPT-OSS 120B (Alta Performance Cognitiva)", "Híbrido (SQL com 120B + Redator com 20B)")
)

# Map da opção visual para a chave real do modelo
model_map = {
    "GPT-OSS 20B (Rápido e Eficiente)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (Alta Performance Cognitiva)": "openai/gpt-oss-120b",
    "Híbrido (SQL com 120B + Redator com 20B)": "hybrid"
}
model_name = model_map[modelo_opcao]

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Dica de Teste:**\n\n"
    "* **GPT-OSS 120B ou Híbrido (Recomendados):** Indicados para consultas complexas. O modelo de 120B possui limites de taxa amplos na Groq, garantindo 100% de estabilidade.\n\n"
    "* **GPT-OSS 20B (Limitações):** Possui cota estrita de tokens/minuto (TPM) na API da Groq. Consultas robustas ao banco de dados que leem esquemas de tabelas podem exceder esse limite temporariamente."
)



# Caixa de texto onde o usuário digita (declarada no nível raiz para flutuar no rodapé)
pergunta = st.chat_input("Pergunte algo (Ex: Qual o status do pedido FCN-1635681090149-01 considerando hoje como 2026-06-15?)")

# Divisão de Abas: Chat interativo e Validador de Cenários
tab_chat, tab_cenarios = st.tabs(["💬 Chat de Atendimento", "🧪 Validador de Cenários (Batch Test)"])

# Função auxiliar para extrair e-mail do cliente
def extrair_email(dados_brutos_str):
    try:
        raw_data = json.loads(dados_brutos_str)
        if isinstance(raw_data, list) and len(raw_data) > 0:
            return raw_data[0].get("e-mail Destinatário", "cliente@teste.com")
    except:
        pass
    return "cliente@teste.com"

# ==========================================
# ABA 1: CHAT DE ATENDIMENTO
# ==========================================
with tab_chat:
    st.subheader("Interação Conversacional em Tempo Real")
    st.markdown("Faça perguntas sobre status de entregas, atrasos ou recusas para ver os agentes em ação.")
    
    # Inicializa o histórico do chat se necessário
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibe as mensagens na tela
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", "\\$"))
            if "pipeline_data" in msg:
                # Exibição organizada dos metadados e agentes
                with st.expander("🔍 Etapa 1: Dados Brutos do Banco (Agente SQL)"):
                    if "query_sql" in msg["pipeline_data"]:
                        st.markdown("**Query SQL Gerada pela IA:**")
                        st.code(msg["pipeline_data"]["query_sql"], language="sql")
                        st.markdown("**Dados Brutos Retornados:**")
                    st.text(msg["pipeline_data"]["dados_brutos"])
                with st.expander("🧠 Etapa 2: Análise Interna & CoT (Agente Analista)"):
                    st.text(msg["pipeline_data"]["resposta_analista"])
                
                tipo_consulta = msg["pipeline_data"].get("tipo_consulta", "STATUS_PEDIDO")
                
                if tipo_consulta == "STATUS_PEDIDO":
                    st.markdown("### 📢 Canais Omnichannel Disparados:")
                    
                    # Renderização dos Cards customizados
                    msg_cliente = msg["pipeline_data"]["mensagem_cliente"]
                    email_dest = extrair_email(msg["pipeline_data"]["dados_brutos"])
                    
                    # Card do WhatsApp
                    st.markdown(f"""
                    <div class="whatsapp-card">
                        <div class="whatsapp-header">📱 WHATSAPP (MENSAGEM AO CLIENTE)</div>
                        <div>{msg_cliente}</div>
                        <div class="whatsapp-footer">Entregue (Double Check) • 17:00 ✓✓</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Card de E-mail
                    st.markdown(f"""
                    <div class="email-card">
                        <div class="email-subject">✉️ Assunto: Atualização sobre a entrega do seu pedido</div>
                        <div class="email-meta">
                            <b>De:</b> sac@ecommerce-tcc.com.br<br>
                            <b>Para:</b> {email_dest}<br>
                            <b>Data:</b> {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
                        </div>
                        <div class="email-body">
                            {msg_cliente.replace('\n', '<br>')}
                        </div>
                        <div class="email-signature">
                            Atenciosamente,<br>
                            <b>Equipe de Atendimento ao Cliente</b><br>
                            Central de Soluções Logísticas E-commerce
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif tipo_consulta == "SEM_DADOS":
                    st.warning("⚠️ **Dados não localizados:** O banco de dados retornou um resultado vazio. Nenhum canal de contato ao cliente (WhatsApp ou E-mail) foi acionado para evitar disparos com erro.")
                else:
                    st.info("ℹ️ **Consulta Administrativa:** Os dados foram extraídos apenas para visualização interna no painel de logística. Nenhum canal de contato de suporte ao cliente (WhatsApp ou E-mail) foi acionado.")
                
                # Card do Slack (Alerta interno logístico - sempre visível)
                st.markdown(f"""
                <div class="slack-card">
                    <div class="slack-header">💬 ALERTA LOGÍSTICA (SLACK #logistica-alertas)</div>
                    <pre style="background: transparent; border: none; font-size: 0.95em; white-space: pre-wrap; margin:0; padding:0; color:#2c2c2c;">{msg["pipeline_data"]["resposta_analista"]}</pre>
                </div>
                """, unsafe_allow_html=True)

    # O processamento da pergunta digitada é feito aqui dentro da tab para renderizar os spinners e respostas no lugar certo
    if pergunta:
        # 1. Adiciona a pergunta do usuário na tela
        st.session_state.messages.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)
            
        # 2. Mostra um spinner e chama o pipeline
        with st.chat_message("assistant"):
            with st.spinner("IA processando dados..."):
                try:
                    if "Multi-Agente" in trilha:
                        historico = st.session_state.messages[:-1]
                        resultados = pipeline_completo(pergunta, trilha_sql="chain", chat_history=historico, model_name=model_name)
                        
                        resposta = f"*(Execução no fluxo **Sistema Multi-Agente Omnichannel** concluída usando o modelo {modelo_opcao}.)*"
                        st.markdown(resposta)
                        
                        # Mostra os expanders informativos
                        with st.expander("🔍 Etapa 1: Dados Brutos do Banco (Agente SQL)"):
                            if "query_sql" in resultados:
                                st.markdown("**Query SQL Gerada pela IA:**")
                                st.code(resultados["query_sql"], language="sql")
                                st.markdown("**Dados Brutos Retornados:**")
                            st.text(resultados["dados_brutos"])
                        with st.expander("🧠 Etapa 2: Análise Interna & CoT (Agente Analista)"):
                            st.text(resultados["resposta_analista"])
                            
                        # Mostra os Cards visuais
                        msg_cliente = resultados["mensagem_cliente"]
                        email_dest = extrair_email(resultados["dados_brutos"])
                        
                        tipo_consulta = resultados.get("tipo_consulta", "STATUS_PEDIDO")
                        
                        if tipo_consulta == "STATUS_PEDIDO":
                            st.markdown("### 📢 Canais Omnichannel Disparados:")
                            
                            st.markdown(f"""
                            <div class="whatsapp-card">
                                <div class="whatsapp-header">📱 WHATSAPP (MENSAGEM AO CLIENTE)</div>
                                <div>{msg_cliente}</div>
                                <div class="whatsapp-footer">Entregue (Double Check) • Recém-enviada ✓✓</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="email-card">
                                <div class="email-subject">✉️ Assunto: Atualização sobre a entrega do seu pedido</div>
                                <div class="email-meta">
                                    <b>De:</b> sac@ecommerce-tcc.com.br<br>
                                    <b>Para:</b> {email_dest}<br>
                                    <b>Data:</b> {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
                                </div>
                                <div class="email-body">
                                    {msg_cliente.replace('\n', '<br>')}
                                </div>
                                <div class="email-signature">
                                    Atenciosamente,<br>
                                    <b>Equipe de Atendimento ao Cliente</b><br>
                                    Central de Soluções Logísticas E-commerce
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif tipo_consulta == "SEM_DADOS":
                            st.warning("⚠️ **Dados não localizados:** O banco de dados retornou um resultado vazio. Nenhum canal de contato ao cliente (WhatsApp ou E-mail) foi acionado para evitar disparos com erro.")
                        else:
                            st.info("ℹ️ **Consulta Administrativa:** Os dados foram extraídos apenas para visualização interna no painel de logística. Nenhum canal de contato de suporte ao cliente (WhatsApp ou E-mail) foi acionado.")
                            
                        # Card do Slack (Alerta interno logístico - sempre visível)
                        st.markdown(f"""
                        <div class="slack-card">
                            <div class="slack-header">💬 ALERTA LOGÍSTICA (SLACK #logistica-alertas)</div>
                            <pre style="background: transparent; border: none; font-size: 0.95em; white-space: pre-wrap; margin:0; padding:0; color:#2c2c2c;">{resultados["resposta_analista"]}</pre>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"{resposta}\n\n{resultados['resposta_analista']}",
                            "pipeline_data": resultados
                        })
                    elif "Autônomo" in trilha:
                        historico = st.session_state.messages[:-1]
                        resposta_bruta = executar_como_agente(pergunta, model_name=model_name, chat_history=historico)
                        st.markdown(str(resposta_bruta).replace("$", "\\$"))
                        info_fluxo = f"\n\n*(Execução no fluxo **Agente de Dados Autônomo (Text-to-SQL)** concluída usando o modelo {modelo_opcao}.)*"
                        st.markdown(info_fluxo)
                        st.session_state.messages.append({"role": "assistant", "content": f"{str(resposta_bruta)}{info_fluxo}"})
                    else:
                        resposta_bruta = executar_como_chain(pergunta, model_name=model_name)
                        resposta = f"**Dados brutos retornados pela Chain de SQL:**\n{resposta_bruta}"
                        st.markdown(str(resposta).replace("$", "\\$"))
                        info_fluxo = f"\n\n*(Execução no fluxo **Pipeline Direto (Apenas Extração SQL)** concluída usando o modelo {modelo_opcao}.)*"
                        st.markdown(info_fluxo)
                        st.session_state.messages.append({"role": "assistant", "content": f"{str(resposta)}{info_fluxo}"})
                    
                except Exception as e:
                    info_fluxo = f"\n\n*(Execução no fluxo **{trilha}** interrompida usando o modelo {modelo_opcao}.)*"
                    erro_msg = f"**Erro durante a execução:** {e}{info_fluxo}"
                    st.error(erro_msg)
                    st.session_state.messages.append({"role": "assistant", "content": erro_msg})

# ==========================================
# ABA 2: AVALIAÇÃO DE CENÁRIOS EM LOTE
# ==========================================
with tab_cenarios:
    st.subheader("🧪 Validação de Cenários em Lote")
    st.markdown("""
        Esta aba tem como objetivo avaliar o desempenho, a acurácia e a estabilidade da arquitetura de IA. 
        Os testes em lote validam a correta classificação de status logísticos e a eficácia das regras de segurança (Guardrails) sobre a base de dados.
    """)
    
    cenarios = [
        {
            "id": 1,
            "nome": "Cenário 1: Entregue com Atraso",
            "pedido": "FCN-1636261090948-01",
            "data_simulada": "2026-06-15",
            "descricao": "O pedido foi entregue após a previsão de entrega do cliente.",
            "pergunta": "Busque todos os dados do pedido FCN-1636261090948-01 na tabela rastreio_intelipost. Considere que a Data Atual de Hoje é 2026-06-15 para realizar sua análise de atraso."
        },
        {
            "id": 2,
            "nome": "Cenário 2: Entregue no Prazo",
            "pedido": "FCN-1636191090792-01",
            "data_simulada": "2026-06-15",
            "descricao": "O pedido foi entregue dentro ou antes do prazo de previsão de entrega.",
            "pergunta": "Busque todos os dados do pedido FCN-1636191090792-01 na tabela rastreio_intelipost. Considere que a Data Atual de Hoje é 2026-06-15 para realizar sua análise de atraso."
        },
        {
            "id": 3,
            "nome": "Cenário 3: Em Trânsito com Atraso",
            "pedido": "FCN-1636631091451-01",
            "data_simulada": "2026-06-15",
            "descricao": "O pedido está a caminho, mas a data prevista (11/06) já passou comparada à data atual.",
            "pergunta": "Busque todos os dados do pedido FCN-1636631091451-01 na tabela rastreio_intelipost. Considere que a Data Atual de Hoje é 2026-06-15 para realizar sua análise de atraso."
        },
        {
            "id": 4,
            "nome": "Cenário 4: Em Trânsito no Prazo",
            "pedido": "FCN-1638181094511-01",
            "data_simulada": "2026-06-15",
            "descricao": "O pedido está a caminho, mas ainda dentro da previsão (17/06) comparada à data atual.",
            "pergunta": "Busque todos os dados do pedido FCN-1638181094511-01 na tabela rastreio_intelipost. Considere que a Data Atual de Hoje é 2026-06-15 para realizar sua análise de atraso."
        },
        {
            "id": 5,
            "nome": "Cenário 5: Guardrail de Pedido Inexistente (Anti-Alucinação)",
            "pedido": "FCN-9999999999999-99",
            "data_simulada": "2026-06-15",
            "descricao": "Simulação de entrada com erro ou código inexistente. O sistema deve acionar o curto-circuito de segurança.",
            "pergunta": "Busque todos os dados do pedido FCN-9999999999999-99 na tabela rastreio_intelipost. Considere que a Data Atual de Hoje é 2026-06-15 para realizar sua análise de atraso."
        }
    ]
    
    st.markdown("### Cenários Cadastrados:")
    for c in cenarios:
        with st.expander(f"📌 {c['nome']}"):
            st.write(f"**Pedido alvo:** `{c['pedido']}`")
            st.write(f"**Data Simulada:** `{c['data_simulada']}`")
            st.write(f"**Descrição do teste:** {c['descricao']}")
            st.info(f"**Pergunta enviada à IA:** \"{c['pergunta']}\"")
            
    if st.button("🚀 Rodar Avaliação em Lote", key="btn_batch_test"):
        resultados_batch = []
        progress_bar = st.progress(0)
        
        for index, c in enumerate(cenarios):
            st.write(f"⏳ Rodando `{c['nome']}` com o modelo `{modelo_opcao}`...")
            start_time = time.time()
            
            try:
                res = pipeline_completo(
                    c["pergunta"], 
                    trilha_sql="chain", 
                    model_name=model_name
                )
                duration = time.time() - start_time
                
                # Tenta extrair a classificação se houver
                classificacao = "N/A"
                for linha in res["resposta_analista"].split("\n"):
                    if "[CLASSIFICAÇÃO]" in linha.upper():
                        classificacao = linha.replace("[CLASSIFICAÇÃO]:", "").replace("[CLASSIFICACAO]:", "").strip()
                        break
                
                resultados_batch.append({
                    "Cenário": c["nome"],
                    "Pedido": c["pedido"],
                    "Status Detectado": classificacao,
                    "Query SQL Gerada": "Executada com sucesso",
                    "Tempo Execução (s)": f"{duration:.2f}s",
                    "Ação Sistêmica": "Mensagem gerada e disparada" if "Sem dados" not in res["resposta_analista"] else "Travada por Guardrail"
                })
            except Exception as e:
                resultados_batch.append({
                    "Cenário": c["nome"],
                    "Pedido": c["pedido"],
                    "Status Detectado": "ERRO",
                    "Query SQL Gerada": "Falha na query",
                    "Tempo Execução (s)": "0.00s",
                    "Ação Sistêmica": f"Abortado com erro: {str(e)}"
                })
            
            progress_bar.progress((index + 1) / len(cenarios))
            time.sleep(3)  # Delay para evitar estourar o limite de requisições por minuto (Rate Limit) da API da Groq
            
        st.success("✅ Testes em Lote finalizados!")
        
        # Cria e exibe o dataframe do relatório
        df_relatorio = pd.DataFrame(resultados_batch)
        st.dataframe(df_relatorio, use_container_width=True)
        
        st.info("📊 **Observação Acadêmica:** Note como no Cenário 4, o status de ação sistêmica indica 'Travada por Guardrail', mostrando que a IA evitou com sucesso alucinar ao receber uma resposta nula do banco.")
