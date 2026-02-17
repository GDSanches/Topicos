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

# Área específica para agrupamento (útil para análise de rede)
TOPICO_AREA = {
    -1: "Não classificado",
    0: "Educação",
    1: "Química",
    2: "Engenharia Elétrica",
    3: "Física",
    4: "Telecomunicações",
    5: "Engenharia Elétrica",
    6: "Ciência da Computação",
    7: "Engenharia Civil",
    8: "Engenharia de Energia",
    9: "Ciência de Alimentos",
    10: "Engenharia Civil",
    11: "Engenharia de Energia",
    12: "Saúde Pública",
    13: "Ciência de Alimentos",
    14: "Engenharia de Materiais",
    15: "Geociências",
    16: "Ecologia",
    17: "Biomedicina",
    18: "Ciência da Computação",
    19: "Educação",
    20: "Comunicação e Sociologia",
    21: "Engenharia Elétrica",
    22: "Ciência da Computação",
    23: "Engenharia Química",
    24: "Medicina",
    25: "Zootecnia",
    26: "Engenharia Elétrica",
    27: "Engenharia de Materiais",
    28: "Engenharia de Energia",
    29: "Engenharia Elétrica",
    30: "Agronomia",
}

# Subárea específica (nível mais granular - ideal para análise de coautoria)
TOPICO_SUBAREA = {
    -1: "Não classificado",
    0: "Políticas Educacionais e Ensino",
    1: "Química de Materiais e Espectroscopia",
    2: "Otimização e Sistemas de Distribuição",
    3: "Física Computacional e Simulação Molecular",
    4: "Redes Móveis e Comunicação Cooperativa",
    5: "Máquinas Elétricas e Acionamentos",
    6: "Computação em Nuvem e Aprendizado de Máquina",
    7: "Geotecnia e Mecânica dos Solos",
    8: "Energia Solar Fotovoltaica",
    9: "Bioquímica de Alimentos e Enzimologia",
    10: "Estruturas de Concreto e Aço",
    11: "Bioenergia e Análise de Viabilidade",
    12: "Epidemiologia e Doenças Crônicas",
    13: "Tecnologia de Óleos e Gorduras",
    14: "Compósitos e Fibras Reforçadas",
    15: "Sensoriamento Remoto e Geoprocessamento",
    16: "Ecologia da Polinização",
    17: "Parasitologia e Imunologia",
    18: "Modelagem e Complexidade Computacional",
    19: "Educação Musical e Cultura",
    20: "Mídias Digitais e Redes Sociais",
    21: "Estabilidade de Sistemas de Potência",
    22: "Mineração de Texto e Classificação",
    23: "Biodiesel e Biocombustíveis",
    24: "Oncologia e Rastreamento de Câncer",
    25: "Aquicultura e Nutrição Animal",
    26: "Conversores de Potência",
    27: "Microfabricação e Dispositivos",
    28: "Sistemas Térmicos e Fotovoltaicos",
    29: "Controle e Automação de Motores",
    30: "Olericultura e Melhoramento Vegetal",
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
df_merged["topico_subarea"] = df_merged["topico"].map(TOPICO_SUBAREA).fillna("Não classificado")
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

# Resumo por área
df_uniq = df_merged.drop_duplicates(subset="id_artigo")
print(f"\nDistribuição por área (artigos únicos):")
resumo_area = df_uniq.groupby("topico_area").size().sort_values(ascending=False)
for area, count in resumo_area.items():
    print(f"  {area}: {count}")

print(f"\nDistribuição por subárea (artigos únicos):")
resumo_sub = df_uniq.groupby("topico_subarea").size().sort_values(ascending=False)
for sub, count in resumo_sub.items():
    print(f"  {sub}: {count}")

print("\nDone!")
