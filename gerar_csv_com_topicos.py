"""
Gera CSV enriquecido com tópicos para análise de redes de coautoria.
Mantém a estrutura original (uma linha por autor/artigo) e adiciona
colunas de tópico para facilitar análise de colaboração entre áreas.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd

CSV_PATH = "c:/Users/Sanches/Documents/Estudos_TCC/Atual-DatasetIFG/Topicos/tabela_final.csv"
TOPICOS_PATH = "c:/Users/Sanches/Documents/Estudos_TCC/Atual-DatasetIFG/Topicos/resultados_topicos/artigos_com_topicos.csv"
PALAVRAS_PATH = "c:/Users/Sanches/Documents/Estudos_TCC/Atual-DatasetIFG/Topicos/resultados_topicos/topicos_palavras.csv"
OUTPUT_PATH = "c:/Users/Sanches/Documents/Estudos_TCC/Atual-DatasetIFG/Topicos/tabela_final_com_topicos.csv"

# Rótulos descritivos para cada tópico (baseado nas palavras-chave)
TOPICO_NOME = {
    -1: "Não classificado",
    0: "Educação e Ciências Sociais",
    1: "Química e Ciência dos Materiais",
    2: "Algoritmos de Otimização e Redes Elétricas",
    3: "Física Computacional e Dinâmica Molecular",
    4: "Redes de Comunicação e 5G",
    5: "Máquinas Elétricas Rotativas",
    6: "Computação em Nuvem e Aprendizado de Máquina",
    7: "Geotecnia e Mecânica dos Solos",
    8: "Energia Solar e Eficiência Energética",
    9: "Ciência de Alimentos (Grãos e Enzimas)",
    10: "Engenharia Estrutural e Concreto",
    11: "Bioenergia e Viabilidade Econômica",
    12: "Saúde Pública e Obesidade",
    13: "Ciência de Alimentos (Lipídios e Óleos)",
    14: "Materiais Compósitos e Fibras",
    15: "Sensoriamento Remoto e Satélites",
    16: "Ecologia e Polinização",
    17: "Parasitologia e Imunologia",
    18: "Modelagem Computacional e Complexidade",
    19: "Educação Musical",
    20: "Interações Sociais e Mídias Digitais",
    21: "Estabilidade de Sistemas de Potência",
    22: "Classificação de Textos e Mineração de Dados",
    23: "Biodiesel e Combustíveis",
    24: "Saúde Pública e Oncologia",
    25: "Aquicultura e Zootecnia",
    26: "Eletrônica de Potência e Conversores",
    27: "Microfabricação e Dispositivos",
    28: "Sistemas Fotovoltaicos e Resfriamento",
    29: "Controle de Motores Elétricos",
    30: "Horticultura e Melhoramento Genético",
}

# Grande área para agrupamento macro (útil para análise de rede)
TOPICO_AREA = {
    -1: "Não classificado",
    0: "Ciências Humanas e Sociais",
    1: "Ciências Exatas e da Terra",
    2: "Engenharias",
    3: "Ciências Exatas e da Terra",
    4: "Engenharias",
    5: "Engenharias",
    6: "Ciências Exatas e da Terra",
    7: "Engenharias",
    8: "Engenharias",
    9: "Ciências Agrárias",
    10: "Engenharias",
    11: "Engenharias",
    12: "Ciências da Saúde",
    13: "Ciências Agrárias",
    14: "Engenharias",
    15: "Ciências Exatas e da Terra",
    16: "Ciências Biológicas",
    17: "Ciências da Saúde",
    18: "Ciências Exatas e da Terra",
    19: "Ciências Humanas e Sociais",
    20: "Ciências Humanas e Sociais",
    21: "Engenharias",
    22: "Ciências Exatas e da Terra",
    23: "Engenharias",
    24: "Ciências da Saúde",
    25: "Ciências Agrárias",
    26: "Engenharias",
    27: "Engenharias",
    28: "Engenharias",
    29: "Engenharias",
    30: "Ciências Agrárias",
}

print("Carregando dados...")

# Carregar tabela original (todas as linhas autor/artigo)
df_original = pd.read_csv(CSV_PATH)
print(f"  Tabela original: {len(df_original)} linhas")

# Carregar atribuição de tópicos (uma linha por artigo)
df_topicos = pd.read_csv(TOPICOS_PATH)[["id_artigo", "topico"]]
print(f"  Artigos com tópico: {len(df_topicos)}")

# Carregar palavras-chave dos tópicos
df_palavras = pd.read_csv(PALAVRAS_PATH)

# Montar coluna de palavras-chave (top 5 por tópico)
palavras_por_topico = (
    df_palavras[df_palavras["rank"] <= 5]
    .groupby("topico")["palavra"]
    .apply(lambda x: "; ".join(x))
    .to_dict()
)

# Merge: adicionar tópico a cada linha da tabela original
df_merged = df_original.merge(df_topicos, on="id_artigo", how="left")

# Adicionar colunas descritivas
df_merged["topico_nome"] = df_merged["topico"].map(TOPICO_NOME).fillna("Não classificado")
df_merged["topico_area"] = df_merged["topico"].map(TOPICO_AREA).fillna("Não classificado")
df_merged["topico_palavras_chave"] = df_merged["topico"].map(palavras_por_topico).fillna("")

# Preencher artigos sem abstract (sem tópico atribuído)
df_merged["topico"] = df_merged["topico"].fillna(-1).astype(int)

# Salvar
df_merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nCSV salvo em: {OUTPUT_PATH}")
print(f"Total de linhas: {len(df_merged)}")
print(f"\nColunas do CSV:")
for col in df_merged.columns:
    print(f"  - {col}")

# Resumo das áreas
print(f"\nDistribuição por grande área (linhas únicas por artigo):")
resumo = (
    df_merged.drop_duplicates(subset="id_artigo")
    .groupby("topico_area")
    .size()
    .sort_values(ascending=False)
)
for area, count in resumo.items():
    print(f"  {area}: {count} artigos")

print("\nDone!")
