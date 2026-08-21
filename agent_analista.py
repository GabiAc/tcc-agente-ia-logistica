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
Sua missão é analisar os dados logísticos do pedido e a pergunta do operador, fornecendo duas mensagens distintas:
1. Uma resposta interna para o OPERADOR (Analista) na terceira pessoa, explicando de forma curta e natural o status do pedido.
2. Uma mensagem de suporte formal destinada ao CLIENTE na segunda pessoa (ex: "Olá [Primeiro Nome], seu pedido..."), adequada para ser disparada por WhatsApp ou E-mail.

PERGUNTA DO OPERADOR:
{pergunta_usuario}

DADOS DO PEDIDO:
{dados_pedido}

CLASSIFICAÇÃO LOGÍSTICA DO SISTEMA:
{classificacao_sistema}

INSTRUÇÕES DE PERSPECTIVA E TOM DE VOZ:
- **[MENSAGEM PARA O OPERADOR] (Interna - Balão do Chat)**:
  - Fale na terceira pessoa sobre o cliente (ex: "O pedido do Diego foi entregue no dia 22/06/2026." ou "O valor do pedido do Diego é R$ 699,00.").
  - Seja extremamente direto, curto e natural, respondendo exatamente à PERGUNTA DO OPERADOR usando as informações logísticas.
  - NÃO saúde o cliente neste campo (não escreva "Olá Diego").
- **[MENSAGEM PARA O CLIENTE] (Externa - WhatsApp/E-mail)**:
  - Fale na segunda pessoa direcionado ao cliente, saudando-o pelo primeiro nome (ex: "Olá Diego, seu pedido...").
  - Se a pergunta do operador for específica (ex: qual o valor, qual a previsão), a mensagem para o cliente deve ser um aviso polido e contextualizado sobre aquela informação específica (ex: "Olá Diego, o valor do seu pedido FCN-1638871095971-01 é R$ 699,00.").
  - Se a pergunta do operador for ampla (ex: qual o status), siga a classificação padrão:
    - 'EM TRÂNSITO NO PRAZO': aviso animador de que o pedido está a caminho.
    - 'EM TRÂNSITO COM ATRASO': desculpas formais e aviso de que está a caminho.
    - 'INTERCORRÊNCIA/EXTRAVIO': aviso de intercorrência, desculpas e que a logística está investigando.
    - 'DEVOLVIDO/FALHA': aviso de falha na entrega, desculpas e que o suporte fará contato para reembolso/reenvio.
    - 'ENTREGUE NO PRAZO' ou 'ENTREGUE COM ATRASO': confirmação de entrega com sucesso.

ATENÇÃO AO MOTIVO DO PROBLEMA (Descrição Transportador ou Motivo Recusa):
- Se os dados indicarem que houve uma Recusa inicial (campo 'Motivo Recusa' presente) mas o status atual no transportador é 'Entregue' ou 'Em Trânsito': você DEVE colocar a recusa no campo 'Detalhe Status' (Ex: "Recusado inicialmente por SEM ESTOQUE") e relatar todo o ocorrido no 'Raciocínio'.

Não utilize emojis em hipótese alguma. Utilize apenas o PRIMEIRO NOME do cliente ao saudá-lo na [MENSAGEM PARA O CLIENTE] (ex: se o nome for 'DIEGO BEZERRA DE SANTANA', escreva apenas 'Diego').

FORMATE SUA RESPOSTA EXATAMENTE ASSIM (NÃO altere o cabeçalho das tags e substitua apenas o texto):
---
[CLASSIFICAÇÃO]: {classificacao_sistema}
[ANÁLISE INTERNA]: Pedido: (Número do Pedido de Venda/Pedido) | Transportadora/Seller: (Nome da Transportadora ou Conta/Seller) | Previsão: (Previsão de Entrega Cliente, caso indisponível use 'N/A') | Status Atual: (Status Transportador) | Detalhe Status: (Descrição Transportador ou 'Recusado inicialmente por [Motivo Recusa]', caso contrário informe "N/A") | Raciocínio: (Resumo em 1 linha detalhando todo o fluxo de forma explicativa para a equipe de logística)

[MENSAGEM PARA O OPERADOR]:
escreva aqui a resposta natural para o operador em 3a pessoa (ex: "O valor do pedido do cliente Diego é R$ 699,00.")

[MENSAGEM PARA O CLIENTE]:
escreva aqui a mensagem polida de suporte direcionada ao cliente em 2a pessoa (ex: "Olá Diego, o valor do seu pedido FCN-1638871095971-01 é R$ 699,00.")
---

Sua resposta final:"""

prompt = PromptTemplate.from_template(template_atendimento)
chain_analista = prompt | llm

def gerar_mensagem_atendimento(dados: dict, classificacao_sistema: str, pergunta_usuario: str = "Qual o status do pedido?", model_name="openai/gpt-oss-20b"):
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
        "classificacao_sistema": classificacao_sistema,
        "pergunta_usuario": pergunta_usuario
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
