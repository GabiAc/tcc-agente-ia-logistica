import pandas as pd
import sqlite3
import os
import glob
import zipfile

# Config
DB_NAME = "banco_tcc.db"
DATA_DIR = r"c:\Users\gabri\OneDrive\Documentos\Pós - BI\TCC - pós"

def load_data_to_sqlite():
    db_path = os.path.join(DATA_DIR, DB_NAME)
    print(f"Conectando ao banco de dados em: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # 1. Intelipost (Rastreio)
    intelipost_files = glob.glob(os.path.join(DATA_DIR, "transactions*.xlsx"))
    if intelipost_files:
        print(f"\nCarregando Intelipost: {os.path.basename(intelipost_files[0])}...")
        df_inteli = pd.read_excel(intelipost_files[0], dtype=str)
        
        # Filtro: Manter apenas pedidos que iniciam com FCN
        df_inteli = df_inteli[df_inteli['Pedido de Venda'].str.startswith('FCN', na=False)]
        
        df_inteli.to_sql('rastreio_intelipost', conn, if_exists='replace', index=False)
        print(f"-> Tabela 'rastreio_intelipost' recriada com {len(df_inteli)} linhas!")

    # 2. Sintese Pedidos
    pedidos_files = glob.glob(os.path.join(DATA_DIR, "pedidos_*.xlsx"))
    if pedidos_files:
        print(f"\nCarregando Síntese Pedidos: {os.path.basename(pedidos_files[0])}...")
        df_pedidos = pd.read_excel(pedidos_files[0], dtype=str)
        
        # Filtro: Manter apenas pedidos que iniciam com FCN
        df_pedidos = df_pedidos[df_pedidos['Pedido'].str.startswith('FCN', na=False)]
        
        df_pedidos.to_sql('sintese_pedidos', conn, if_exists='replace', index=False)
        print(f"-> Tabela 'sintese_pedidos' recriada com {len(df_pedidos)} linhas!")

    # 3. Sintese Recusas
    recusas_files = glob.glob(os.path.join(DATA_DIR, "pedidosprodutosrecusados*.xlsx"))
    if recusas_files:
        print(f"\nCarregando Síntese Recusas: {os.path.basename(recusas_files[0])}...")
        df_recusas = pd.read_excel(recusas_files[0], dtype=str)
        
        # Filtro: Manter apenas pedidos que iniciam com FCN
        df_recusas = df_recusas[df_recusas['Pedido'].str.startswith('FCN', na=False)]
        
        df_recusas.to_sql('sintese_recusas', conn, if_exists='replace', index=False)
        print(f"-> Tabela 'sintese_recusas' recriada com {len(df_recusas)} linhas!")

    # 4. VTEX (Report.zip -> Report.csv)
    vtex_files = glob.glob(os.path.join(DATA_DIR, "Report*.zip"))
    if vtex_files:
        vtex_zip = vtex_files[0]
        print(f"\nCarregando VTEX do arquivo: {os.path.basename(vtex_zip)}...")
        try:
            with zipfile.ZipFile(vtex_zip, 'r') as z:
                csv_name = z.namelist()[0]
                print(f"Lendo {csv_name} de dentro do ZIP...")
                df_vtex = pd.read_csv(z.open(csv_name), sep=';', low_memory=False, dtype=str)
                df_vtex.to_sql('pedidos_vtex', conn, if_exists='replace', index=False)
                print(f"-> Tabela 'pedidos_vtex' recriada com {len(df_vtex)} linhas!")
        except Exception as e:
            print(f"Erro ao carregar VTEX: {e}")

    conn.close()
    print("\nBanco de dados atualizado com sucesso!")

if __name__ == "__main__":
    load_data_to_sqlite()
