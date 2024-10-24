import pandas as pd
import numpy as np

def create_lotofacil_onehot(csv_file):
    """
    Cria one hot encoding para dados da Lotofácil.
    
    Parameters:
    csv_file (str): Caminho do arquivo CSV com os dados da Lotofácil
    
    Returns:
    pandas.DataFrame: DataFrame com o one hot encoding
    """
    # Lê o arquivo CSV
    df = pd.read_csv(csv_file)
    
    # Cria matriz vazia para one hot encoding (n_sorteios x 25 números possíveis)
    n_sorteios = len(df)
    onehot_matrix = np.zeros((n_sorteios, 25))
    
    # Para cada linha (sorteio)
    for idx, row in df.iterrows():
        # Pega os números sorteados (converte para int e subtrai 1 para índice 0-based)
        numeros_sorteados = row.values.astype(int) - 1
        # Marca 1 nas posições dos números sorteados
        onehot_matrix[idx, numeros_sorteados] = 1
    
    # Cria DataFrame com o one hot encoding
    colunas = [f'num_{i+1}' for i in range(25)]
    onehot_df = pd.DataFrame(onehot_matrix, columns=colunas)
    
    return onehot_df

# Exemplo de uso
if __name__ == "__main__":
    # Substitua pelo caminho do seu arquivo CSV
    arquivo_csv = "a.csv"
    
    # Gera one hot encoding
    resultado = create_lotofacil_onehot(arquivo_csv)
    
    # Mostra as primeiras linhas
    print("\nPrimeiras linhas do one hot encoding:")
    print(resultado.head())
    
    # Salva em um novo arquivo CSV
    resultado.to_csv("lotofacil_onehot.csv", index=False)
    print("\nOne hot encoding salvo em 'lotofacil_onehot.csv'")