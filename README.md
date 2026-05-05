# 🚀 Geração de Embeddings com Ollama e Python

Este projeto demonstra como instalar e utilizar o Ollama localmente para gerar vetores de embeddings a partir de um texto simples utilizando Python.

---

## 📌 Objetivo

O objetivo deste projeto é:

- Instalar e configurar o Ollama localmente
- Utilizar um modelo de embeddings
- Gerar vetores numéricos a partir de um texto
- Exibir informações sobre o vetor gerado

---

## 🧩 Tecnologias Utilizadas

- Python 3.x
- Ollama (execução local de modelos de IA)
- Modelo: `nomic-embed-text`

---

## ⚙️ Pré-requisitos

Antes de começar, você precisa ter instalado:

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)
- Curl (para instalar o Ollama)

---

## 🐧 Instalação do Ollama

Execute o comando abaixo no terminal:

```bash
curl -fsSL https://ollama.com/install.sh | sh

Após a instalação, inicie o serviço:

ollama serve
📦 Download do modelo de embeddings

Baixe o modelo necessário:

ollama pull nomic-embed-text
🐍 Instalação das dependências Python

Instale a biblioteca oficial do Ollama:

pip install ollama
▶️ Execução do Script

Execute o script Python:

python main.py
🧪 Exemplo de Código
import ollama

texto = "Rodando ollama localmente com sucesso."

response = ollama.embeddings(
    model="nomic-embed-text",
    prompt=texto
)

embedding = response["embedding"]

print("Texto:", texto)
print("Tamanho do vetor:", len(embedding))
print("Primeiros valores:", embedding[:10])
📊 Saída Esperada
Texto: O Ollama permite rodar modelos de IA localmente.
Tamanho do vetor: 768
Primeiros valores: [0.0123, -0.0345, 0.0987, ...]
⚠️ Possíveis Problemas
❌ Ollama não está rodando

Verifique com:

curl http://localhost:11434
❌ Modelo não encontrado

Execute novamente:

ollama pull nomic-embed-text
📚 Referências
https://ollama.com
https://github.com/ollama/ollama
👨‍💻 Autor

Projeto desenvolvido para fins acadêmicos.
