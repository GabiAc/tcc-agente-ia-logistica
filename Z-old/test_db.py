import sqlite3
import pandas as pd

conn = sqlite3.connect('banco_tcc.db')
c = conn.cursor()

print("Pedidos não entregues na rastreio_intelipost:")
try:
    df = pd.read_sql("SELECT \"Pedido de Venda\", \"Status Transportador\", \"Data do último status\", \"Previsão Entrega Cliente\" FROM rastreio_intelipost WHERE \"Status Transportador\" != 'Entregue' LIMIT 5", conn)
    print(df)
except Exception as e:
    print(e)

