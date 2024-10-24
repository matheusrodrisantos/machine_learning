import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LotofacilAnalyzer:
    def __init__(self, csv_file, min_support=0.3, min_confidence=0.5):
        """
        Inicializa o analisador da Lotofácil
        """
        self.csv_file = csv_file
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.df = None
        self.onehot_df = None
        self.frequent_itemsets = None
        self.rules = None
        self.recommended_numbers = None
        
    def load_and_prepare_data(self):
        """
        Carrega e prepara os dados para análise
        """
        # Lê o arquivo CSV
        self.df = pd.read_csv(self.csv_file)
        
        # Cria one hot encoding de forma otimizada
        n_sorteios = len(self.df)
        onehot_matrix = np.zeros((n_sorteios, 25))
        
        # Usa numpy para processamento mais rápido
        for idx, row in enumerate(self.df.values):
            numeros_sorteados = row.astype(int) - 1
            onehot_matrix[idx, numeros_sorteados] = 1
        
        # Cria DataFrame com one hot encoding
        self.onehot_df = pd.DataFrame(
            onehot_matrix,
            columns=[f'num_{i+1}' for i in range(25)]
        )
        
    def analyze_patterns(self):
        """
        Executa a análise de padrões usando Apriori
        """
        # Encontra itemsets frequentes com processamento otimizado
        self.frequent_itemsets = apriori(
            self.onehot_df,
            min_support=self.min_support,
            use_colnames=True,
            max_len=6,  # Aumentado para 6 números
            verbose=1
        )
        
        # Gera regras de associação
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        
    def calculate_recommendations(self):
        """
        Calcula recomendações de números baseadas em múltiplos critérios
        """
        # Análise de frequência básica
        numero_frequency = self.onehot_df.sum() / len(self.onehot_df)
        
        # Inicializa scores
        number_scores = {i+1: 0 for i in range(25)}
        
        # 1. Pontuação baseada na frequência individual
        for i in range(25):
            number_scores[i+1] += numero_frequency[f'num_{i+1}'] * 0.3  # Peso de 30%
            
        # 2. Pontuação baseada nas regras de associação
        def extract_numbers_from_frozenset(fs):
            return [int(str(item).split('_')[1]) for item in fs]
        
        for _, rule in self.rules.iterrows():
            ant_nums = extract_numbers_from_frozenset(rule['antecedents'])
            cons_nums = extract_numbers_from_frozenset(rule['consequents'])
            
            score = rule['confidence'] * rule['support'] * 0.4  # Peso de 40%
            for num in ant_nums + cons_nums:
                number_scores[num] += score
                
        # 3. Análise de tendências recentes (últimos 10 jogos)
        recent_games = self.onehot_df.iloc[-10:]
        recent_freq = recent_games.sum() / len(recent_games)
        
        for i in range(25):
            number_scores[i+1] += recent_freq[f'num_{i+1}'] * 0.3  # Peso de 30%
        
        # Seleciona os números mais promissores
        self.recommended_numbers = sorted(
            number_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        
    def analyze_number_patterns(self):
        """
        Analisa padrões específicos nos números sorteados
        """
        results = {
            'pares_impares': [],
            'soma_total': [],
            'quadrantes': []
        }
        
        for _, row in self.df.iterrows():
            numeros = row.values.astype(int)
            
            # Conta pares e ímpares
            pares = sum(1 for n in numeros if n % 2 == 0)
            results['pares_impares'].append((pares, 15-pares))
            
            # Soma total
            results['soma_total'].append(sum(numeros))
            
            # Análise por quadrantes
            quadrantes = [0] * 4
            for n in numeros:
                if n <= 6:
                    quadrantes[0] += 1
                elif n <= 12:
                    quadrantes[1] += 1
                elif n <= 18:
                    quadrantes[2] += 1
                else:
                    quadrantes[3] += 1
            results['quadrantes'].append(quadrantes)
            
        return results
    
    def print_comprehensive_analysis(self):
        """
        Imprime análise completa dos resultados
        """
        print("\n=== ANÁLISE COMPLETA DA LOTOFÁCIL ===")
        print(f"Total de jogos analisados: {len(self.df)}")
        
        # 1. Números Recomendados
        print("\n1. NÚMEROS RECOMENDADOS PARA O PRÓXIMO JOGO:")
        numbers_only = sorted([num for num, _ in self.recommended_numbers])
        print(f"Números: {numbers_only}")
        
        # 2. Análise de Padrões
        patterns = self.analyze_number_patterns()
        
        print("\n2. ANÁLISE DE PADRÕES:")
        
        # 2.1 Distribuição Pares/Ímpares
        pares_count = Counter(tuple(p) for p in patterns['pares_impares'])
        most_common_par_impar = max(pares_count.items(), key=lambda x: x[1])
        print(f"\nDistribuição mais comum de Pares/Ímpares:")
        print(f"Pares: {most_common_par_impar[0][0]}, Ímpares: {most_common_par_impar[0][1]}")
        print(f"Ocorrências: {most_common_par_impar[1]} jogos")
        
        # 2.2 Soma Total
        soma_media = np.mean(patterns['soma_total'])
        soma_std = np.std(patterns['soma_total'])
        print(f"\nSoma dos números:")
        print(f"Média: {soma_media:.1f}")
        print(f"Faixa mais comum: {soma_media-soma_std:.1f} a {soma_media+soma_std:.1f}")
        
        # 2.3 Distribuição por Quadrantes
        quadrantes_media = np.mean(patterns['quadrantes'], axis=0)
        print(f"\nMédia de números por quadrante:")
        for i, media in enumerate(quadrantes_media, 1):
            print(f"Quadrante {i}: {media:.1f} números")
        
        # 3. Top Regras de Associação
        print("\n3. REGRAS DE ASSOCIAÇÃO MAIS FORTES:")
        top_rules = self.rules.sort_values('confidence', ascending=False).head()
        for _, rule in top_rules.iterrows():
            ant = sorted([int(str(item).split('_')[1]) for item in rule['antecedents']])
            cons = sorted([int(str(item).split('_')[1]) for item in rule['consequents']])
            print(f"\nSe aparecem {ant}")
            print(f"Então frequentemente aparecem {cons}")
            print(f"Confiança: {rule['confidence']:.3f}")
            print(f"Suporte: {rule['support']:.3f}")

def main():
    # Configurações
    arquivo_csv = "lotofacil_onehot.csv"
    
    # Inicializa e executa análise
    print("Iniciando análise...")
    start_time = datetime.now()
    
    analyzer = LotofacilAnalyzer(
        arquivo_csv,
        min_support=0.25,  # Ajustado para dados maiores
        min_confidence=0.45
    )
    
    analyzer.load_and_prepare_data()
    analyzer.analyze_patterns()
    analyzer.calculate_recommendations()
    analyzer.print_comprehensive_analysis()
    
    end_time = datetime.now()
    print(f"\nTempo total de execução: {end_time - start_time}")

if __name__ == "__main__":
    main()