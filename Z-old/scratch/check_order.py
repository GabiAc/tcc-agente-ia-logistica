import sqlite3
conn = sqlite3.connect("banco_tcc.db")
cursor = conn.cursor()
cursor.execute("SELECT [Pedido de Venda], [Data Entrega], [Previsão Entrega Cliente] FROM rastreio_intelipost WHERE [Status Transportador] = 'Entregue'")
rows = cursor.fetchall()
found = False
for r in rows:
    ped, ent, prev = r
    if ent and prev:
        # compare dates YYYY-MM-DD
        ent_dt = ent.split(" ")[0]
        prev_dt = prev.split(" ")[0]
        if ent_dt > prev_dt:
            print(f"FOUND LATE DELIVERED ORDER: Pedido={ped}, Delivery={ent}, Forecast={prev}")
            found = True
            break
if not found:
    print("NO LATE DELIVERED ORDER FOUND!")
conn.close()
