import sqlite3
conn = sqlite3.connect('banco_tcc.db')
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT "Status Transportador" FROM rastreio_intelipost')
print(cursor.fetchall())
