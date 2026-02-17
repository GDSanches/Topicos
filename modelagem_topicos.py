"""
Modelagem de Tópicos - Artigos IFG
Utiliza BERTopic com sentence-transformers para modelagem de tópicos
sobre os abstracts dos artigos.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================================
print("=" * 60)
print("MODELAGEM DE TÓPICOS - ARTIGOS IFG")
print("=" * 60)

CSV_PATH = "c:/Users/Sanches/Documents/Estudos_TCC/Atual-DatasetIFG/Topicos/tabela_final.csv"
OUTPUT_DIR = "c:/Users/Sanches/Documents/Estudos_TCC/Atual-DatasetIFG/Topicos/resultados_topicos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Deduplicar por id_artigo (cada artigo aparece múltiplas vezes, uma por autor)
df_unique = df.drop_duplicates(subset="id_artigo").copy()
print(f"\nTotal de artigos únicos: {len(df_unique)}")

# Remover abstracts vazios
df_unique = df_unique.dropna(subset=["api_abstract"]).copy()
df_unique = df_unique[df_unique["api_abstract"].str.strip() != ""].copy()
df_unique = df_unique.reset_index(drop=True)
print(f"Artigos com abstract válido: {len(df_unique)}")

docs = df_unique["api_abstract"].tolist()

# ============================================================
# 2. CONFIGURAR E TREINAR O MODELO BERTopic
# ============================================================
print("\n[1/3] Carregando modelo de embeddings (multilingual)...")

# Modelo multilingual para lidar com abstracts em PT e EN
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Stopwords em português e inglês para o CountVectorizer
stopwords_pt = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com",
    "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
    "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
    "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo",
    "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo",
    "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você",
    "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha",
    "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós",
    "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse",
    "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu",
    "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela",
    "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles",
    "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão",
    "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos",
    "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam",
    "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos",
    "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos",
    "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam",
    "houvesse", "houvéssemos", "houvessem", "houver", "houvermos",
    "houverem", "houverei", "houverá", "houveremos", "houverão",
    "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "era",
    "éramos", "eram", "fui", "foi", "fomos", "foram", "fora", "fôramos",
    "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for",
    "formos", "forem", "serei", "será", "seremos", "serão", "seria",
    "seríamos", "seriam", "tenho", "tem", "temos", "tém", "tinha",
    "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera",
    "tivéramos", "tenha", "tenhamos", "tenham", "tivesse", "tivéssemos",
    "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos",
    "terão", "teria", "teríamos", "teriam", "sobre", "partir", "ainda",
    "assim", "pode", "podem", "onde", "bem", "cada", "dois", "três",
    "forma", "uso", "trabalho", "estudo", "resultados", "dados", "sendo",
    "além", "através", "outros", "outras", "outro", "outra", "todo",
    "toda", "todos", "todas", "são", "foram", "entre", "desde", "então",
    "durante", "antes", "apenas", "porque", "porém", "quanto", "enquanto",
    "contra", "dentro", "sempre", "nesse", "nessa", "nesses", "nessas",
    "neste", "nesta", "nestes", "nestas", "desse", "dessa", "desses",
    "dessas", "deste", "desta", "destes", "destas", "naquele", "naquela",
    "daquele", "daquela"
]

stopwords_en = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "where", "when", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "if", "then", "also",
    "about", "up", "out", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further", "once",
    "here", "there", "any", "while", "however", "well", "our", "results",
    "using", "based", "also", "study", "paper", "proposed", "approach"
]

all_stopwords = list(set(stopwords_pt + stopwords_en))

# Vectorizer com stopwords customizadas
vectorizer_model = CountVectorizer(
    stop_words=all_stopwords,
    min_df=3,
    ngram_range=(1, 2)
)

# c-TF-IDF com redução de frequências globais
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

print("[2/3] Gerando embeddings dos abstracts...")
embeddings = embedding_model.encode(docs, show_progress_bar=True, batch_size=32)

print("[3/3] Treinando modelo BERTopic...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    language="multilingual",
    nr_topics="auto",
    min_topic_size=10,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings)

# ============================================================
# 3. RESULTADOS
# ============================================================
print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

topic_info = topic_model.get_topic_info()
print(f"\nNúmero de tópicos encontrados: {len(topic_info) - 1}")  # -1 para excluir outliers (-1)
print(f"Documentos classificados como outlier (tópico -1): {(np.array(topics) == -1).sum()}")
print(f"\nTop tópicos por frequência:")
print(topic_info.head(20).to_string(index=False))

# Palavras-chave de cada tópico
print("\n\nPalavras-chave por tópico:")
for topic_id in topic_info["Topic"].values:
    if topic_id == -1:
        continue
    words = topic_model.get_topic(topic_id)
    top_words = ", ".join([w for w, _ in words[:8]])
    count = topic_info[topic_info["Topic"] == topic_id]["Count"].values[0]
    print(f"  Tópico {topic_id:3d} ({count:4d} docs): {top_words}")

# ============================================================
# 4. SALVAR RESULTADOS
# ============================================================
print(f"\n\nSalvando resultados em: {OUTPUT_DIR}")

# Salvar info dos tópicos
topic_info.to_csv(f"{OUTPUT_DIR}/topicos_info.csv", index=False)

# Salvar atribuição de tópico por artigo
df_unique = df_unique.copy()
df_unique["topico"] = topics
if probs is not None and isinstance(probs, np.ndarray) and probs.ndim == 2:
    df_unique["topico_prob"] = [p.max() if len(p) > 0 else 0.0 for p in probs]
else:
    df_unique["topico_prob"] = 0.0
df_unique[["id_artigo", "titulo", "ano", "api_abstract", "topico", "topico_prob"]].to_csv(
    f"{OUTPUT_DIR}/artigos_com_topicos.csv", index=False
)

# Salvar palavras-chave de cada tópico
topic_words_list = []
for topic_id in topic_info["Topic"].values:
    if topic_id == -1:
        continue
    words = topic_model.get_topic(topic_id)
    for rank, (word, score) in enumerate(words):
        topic_words_list.append({
            "topico": topic_id,
            "rank": rank + 1,
            "palavra": word,
            "score": round(score, 4)
        })
pd.DataFrame(topic_words_list).to_csv(f"{OUTPUT_DIR}/topicos_palavras.csv", index=False)

# Gerar visualizações (salvar como HTML)
print("Gerando visualizações...")
try:
    fig_topics = topic_model.visualize_barchart(top_n_topics=15, n_words=8)
    fig_topics.write_html(f"{OUTPUT_DIR}/viz_barchart.html")
    print("  - viz_barchart.html")
except Exception as e:
    print(f"  - Erro no barchart: {e}")

try:
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html(f"{OUTPUT_DIR}/viz_hierarquia.html")
    print("  - viz_hierarquia.html")
except Exception as e:
    print(f"  - Erro na hierarquia: {e}")

try:
    fig_heatmap = topic_model.visualize_heatmap()
    fig_heatmap.write_html(f"{OUTPUT_DIR}/viz_heatmap.html")
    print("  - viz_heatmap.html")
except Exception as e:
    print(f"  - Erro no heatmap: {e}")

try:
    fig_docs = topic_model.visualize_documents(docs, embeddings=embeddings)
    fig_docs.write_html(f"{OUTPUT_DIR}/viz_documentos.html")
    print("  - viz_documentos.html")
except Exception as e:
    print(f"  - Erro na visualização de docs: {e}")

# Tópicos ao longo do tempo
try:
    timestamps = df_unique["ano"].tolist()
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topics_over_time.to_csv(f"{OUTPUT_DIR}/topicos_ao_longo_tempo.csv", index=False)
    fig_time = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
    fig_time.write_html(f"{OUTPUT_DIR}/viz_topicos_tempo.html")
    print("  - viz_topicos_tempo.html")
except Exception as e:
    print(f"  - Erro nos tópicos ao longo do tempo: {e}")

# Salvar modelo
topic_model.save(f"{OUTPUT_DIR}/modelo_bertopic", serialization="safetensors", save_ctfidf=True)
print(f"  - Modelo salvo em: {OUTPUT_DIR}/modelo_bertopic")

print("\n" + "=" * 60)
print("MODELAGEM CONCLUÍDA COM SUCESSO!")
print("=" * 60)
