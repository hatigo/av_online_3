import ollama

texto = "Rodando ollama localmente com sucesso."

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=texto
)

embedding = response["embedding"]

print("Texto:", texto)
print("\nTamanho do vetor:", len(embedding))
print("\nPrimeiros 10 valores do vetor:")
print(embedding[:10])
