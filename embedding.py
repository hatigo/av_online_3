import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, normalize
import matplotlib.pyplot as plt
from langchain_community.embeddings import OllamaEmbeddings


# ==============================
# CONFIGURAÇÃO
# ==============================
logging.basicConfig(level=logging.INFO)
MODEL_NAME = "mxbai-embed-large"


# ==============================
# EMBEDDINGS SERVICE
# ==============================
class EmbeddingService:
    def __init__(self, model_name: str):
        self.embeddings = OllamaEmbeddings(model=model_name)

    def embed_documents(self, docs):
        logging.info("Gerando embeddings dos documentos...")
        vectors = self.embeddings.embed_documents(docs)
        return normalize(np.array(vectors))

    def embed_query(self, query):
        logging.info("Gerando embedding da query...")
        vector = self.embeddings.embed_query(query)
        return normalize([vector])[0]


# ==============================
# BUSCA SEMÂNTICA
# ==============================
def semantic_search(query, documents, embeddings_matrix, service, top_k=3):
    query_vector = service.embed_query(query)
    scores = cosine_similarity([query_vector], embeddings_matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(scores[idx]),
            "document": documents[idx]
        })

    return results


# ==============================
# VISUALIZAÇÃO COM TSNE
# ==============================
def visualize_embeddings(words, categories, embeddings_matrix):
    logging.info("Reduzindo dimensionalidade com t-SNE...")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_matrix)

    df = pd.DataFrame({
        "Palavra": words,
        "Categoria": categories,
        "X": embeddings_2d[:, 0],
        "Y": embeddings_2d[:, 1]
    })

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(df["Categoria"])

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df["X"], df["Y"], c=encoded, cmap="viridis", alpha=0.7)

    # legenda segura
    for label in label_encoder.classes_:
        plt.scatter([], [], label=label)
    plt.legend(title="Categorias")

    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.title("Visualização dos Embeddings (t-SNE)")
    plt.show()


# ==============================
# MAIN
# ==============================
def main():
    service = EmbeddingService(MODEL_NAME)

    # --------------------------
    # PARTE 1: BUSCA SEMÂNTICA
    # --------------------------
    documents_python = [
        "Variáveis em Python e os tipos de dados básicos como int, float e string.",
        "Estruturas de controle em Python: if, else, e elif para tomada de decisões.",
        "Loops em Python, como for e while, para repetição de tarefas.",
        "Funções em Python: como definir e chamar funções usando a palavra-chave def."
    ]

    doc_embeddings = service.embed_documents(documents_python)

    query = "Como criar funções em Python?"
    results = semantic_search(query, documents_python, doc_embeddings, service)

    print("\n🔎 Resultados da busca:")
    for r in results:
        print(f"{r['score']:.4f} -> {r['document']}")

    # --------------------------
    # PARTE 2: VISUALIZAÇÃO
    # --------------------------
    palavras = [
        "maçã","banana","laranja","uva","abacaxi","morango","kiwi","melancia","cereja","manga",
        "pêssego","ameixa","pera","framboesa","groselha","abacate","figo","limão","coco","tangerina",
        "caqui","caju","açaí","graviola","jabuticaba","maracujá","pinha","carambola","pitanga","romã",
        "carro","bicicleta","avião","trem","moto","ônibus","barco","patinete","navio","metrô",
        "caminhão","helicóptero","trator","balão","espaçonave","skate","scooter","jato","táxi","hoverboard",
        "triciclo","trólebus","furgão","carreta","veleiro","caiaque","caminhonete","quadriciclo","buggy","limusine",
        "futebol","basquete","natação","vôlei","tênis","corrida","ciclismo","golfe","boxe","rugby",
        "hóquei","beisebol","críquete","polo","esgrima","surf","judô","karatê","badminton","ginástica",
        "skate","snowboard","esqui","mergulho","remo","caminhada","triatlo","halterofilismo","ioga","pilates",
        "felicidade","tristeza","raiva","amor","medo","surpresa","ansiedade","esperança","frustração","confiança",
        "empatia","gratidão","desespero","nostalgia","ciúmes","satisfação","alívio","arrependimento","euforia","vergonha",
        "culpa","orgulho","inveja","solidão","tédio","excitação","admiração","curiosidade","pânico","respeito"
    ]

    categorias = (
        ["fruta"] * 30 +
        ["transporte"] * 30 +
        ["esporte"] * 30 +
        ["emoção"] * 30
    )

    embeddings_words = service.embed_documents(palavras)

    visualize_embeddings(palavras, categorias, embeddings_words)


if __name__ == "__main__":
    main()