import sqlite3
import pandas as pd
from integracao_fase4 import pipeline_completo

def testar_cenario(nome_cenario, pedido, data_atual_mock):
    print(f"\n{'='*50}")
    print(f"🧪 TESTANDO CENÁRIO: {nome_cenario}")
    print(f"{'='*50}")
    pergunta = f"Busque todos os dados do pedido {pedido} na tabela rastreio_intelipost. Considere que a Data Atual de Hoje é {data_atual_mock} para realizar sua análise de atraso."
    pipeline_completo(pergunta, trilha_sql="chain")

if __name__ == "__main__":
    # Cenário 1: Entregue no Prazo
    # Pedido entregue no dia 10/06, previsão era 05/06. Opa, isso é ENTREGUE COM ATRASO!
    # Vamos rodar com data atual 15/06 pra ver a reação de entregue com atraso.
    testar_cenario("ENTREGUE COM ATRASO", "FCN-1636191090792-01", "2026-06-15")

    # Cenário 2: Em trânsito no prazo
    # Pedido não entregue, previsão é 16/06. Vamos simular que hoje é 05/06.
    testar_cenario("EM TRÂNSITO (NO PRAZO)", "FCN-1636221090871-01", "2026-06-05")

    # Cenário 3: Em trânsito com atraso
    # Mesmo pedido, mas vamos simular que hoje é 20/06 (já passou a previsão que era 16/06).
    testar_cenario("EM TRÂNSITO (COM ATRASO)", "FCN-1636221090871-01", "2026-06-20")
