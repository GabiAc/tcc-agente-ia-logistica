import os
import sys
import io


from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# ==========================================
# 1. CONFIGURAÇÃO DO MODELO
# ==========================================
# Vamos continuar usando o GPT OSS 20B da Groq, que é super rápido e ótimo para gerar textos curtos
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3) # Temperatura em 0.3 para a mensagem ficar mais "humana/empática" sem inventar fatos

# ==========================================
# 2. PROMPT CHAIN-OF-THOUGHT (CoT) PARA ATENDIMENTO
# ==========================================
template_atendimento = """Você é um Agente de Atendimento ao Cliente de E-commerce sênior.
Sua missão é receber os dados logísticos de um pedido, a classificação de status já determinada pelo sistema e escrever uma mensagem humanizada para o cliente.

DADOS DO PEDIDO:
{dados_pedido}

CLASSIFICAÇÃO LOGÍSTICA DO SISTEMA:
{classificacao_sistema}

INSTRUÇÕES DE TOM DE VOZ E MENSAGEM:
- Se a classificação for 'EM TRÂNSITO NO PRAZO', use um tom informativo e animador, avisando que o pedido está a caminho.
- Se a classificação for 'EM TRÂNSITO COM ATRASO', peça desculpas sinceras pelo atraso e informe que o pedido está a caminho.
- Se a classificação for 'INTERCORRÊNCIA/EXTRAVIO', informe que houve uma intercorrência no fluxo de entrega, peça desculpas e informe que a equipe de logística está investigando o caso ativamente para resolver.
- Se a classificação for 'DEVOLVIDO/FALHA', informe explicitamente que houve um problema/falha na tentativa de entrega (ou que foi devolvido/cancelado/recusado), peça desculpas sinceras e informe que o suporte entrará em contato para organizar o reenvio ou reembolso. NUNCA diga que o pedido está "atrasado" ou "a caminho".
- Se a classificação for 'ENTREGUE NO PRAZO' ou 'ENTREGUE COM ATRASO', use um tom de celebração confirmando que foi entregue.

ATENÇÃO AO MOTIVO DO PROBLEMA (Descrição Transportador ou Motivo Recusa):
- Se os dados indicarem que houve uma Recusa inicial (campo 'Motivo Recusa' presente) mas o status atual no transportador é 'Entregue' ou 'Em Trânsito': você DEVE colocar a recusa no campo 'Detalhe Status' (Ex: "Recusado inicialmente por SEM ESTOQUE") e relatar todo o ocorrido de forma explicativa no 'Raciocínio' (Ex: "Pedido recusado inicialmente pelo seller X por falta de estoque, mas faturado posteriormente por outro seller e entregue em DD/MM/YYYY").
- Se for entrega normal, o campo 'Descrição Transportador' contém o detalhamento da ocorrência. Se o campo for nulo ou vazio e não houver recusa, trate-o apenas como um problema geral.

Não utilize emojis em hipótese alguma. Importante: utilize apenas o PRIMEIRO NOME do cliente ao saudá-lo na mensagem (ex: se o nome for 'LUIS FELIPE BECK GIARDULLO', escreva apenas 'Luis' com inicial maiúscula; se for 'MARIA DA SILVA', escreva 'Maria').

FORMATE SUA RESPOSTA EXATAMENTE ASSIM (NÃO altere o cabeçalho das tags e substitua apenas o texto):
---
[CLASSIFICAÇÃO]: {classificacao_sistema}
[ANÁLISE INTERNA]: Pedido: (Número do Pedido de Venda/Pedido) | Transportadora/Seller: (Nome da Transportadora ou Conta/Seller) | Previsão: (Previsão de Entrega Cliente, caso indisponível use a Data do Pedido ou 'N/A') | Status Atual: (Status Transportador, ou se for recusa sem faturamento use 'Recusado') | Detalhe Status: (Descrição Transportador ou 'Recusado inicialmente por [Motivo Recusa]' se houver recusa, caso contrário informe "N/A") | Raciocínio: (Resumo em 1 linha detalhando todo o fluxo de forma explicativa para a equipe de logística, ex: detalhando a recusa inicial por falta de estoque e o faturamento/entrega posterior)

[MENSAGEM PARA O CLIENTE]:
escreva aqui a mensagem final de WhatsApp sem usar nenhum emoji
---

Sua resposta final:"""

prompt = PromptTemplate.from_template(template_atendimento)
chain_analista = prompt | llm

def gerar_mensagem_atendimento(dados: dict, classificacao_sistema: str, model_name="openai/gpt-oss-20b"):
    if "llama-3.1-8b" in model_name:
        model_name = "openai/gpt-oss-20b"
    elif "llama-3.3-70b" in model_name:
        model_name = "openai/gpt-oss-120b"
        
    print(f"\n[Agente Analista] Raciocinando sobre os dados e escrevendo o texto com {model_name}...")
    
    # Transforma o dicionário em texto formatado para o LLM ler facilmente
    dados_str = "\n".join([f"- {k}: {v}" for k, v in dados.items()])
    
    from langchain_groq import ChatGroq
    local_llm = ChatGroq(model=model_name, temperature=0.3)
    local_chain = prompt | local_llm
    
    resposta = local_chain.invoke({
        "dados_pedido": dados_str,
        "classificacao_sistema": classificacao_sistema
    })
    return resposta.content

if __name__ == "__main__":
    # Criamos um cenário fictício baseado nas colunas que vimos no seu banco
    dados_teste_atraso = {
        "Nome do Cliente": "Luiz Carlos Marques",
        "Numero do Pedido": "FCN-1513080904883",
        "Status Transportadora": "Pendente na base",
        "Previsao de Entrega": "10/05/2026",
        "Data Atual Simulada": "19/05/2026", # Já se passaram 9 dias!
        "Transportadora": "Loggi Express"
    }
    
    print("===== TESTE FASE 3: CENÁRIO DE ATRASO =====")
    try:
        resultado = gerar_mensagem_atendimento(dados_teste_atraso)
        print(f"\n{resultado}\n")
    except Exception as e:
        print(f"Erro na execução do Agente Analista: {e}")
