# Mini RAG — Company Policy Assistant

A Python-based Retrieval-Augmented Generation (RAG) application that answers questions about company policies using local documents.

The project combines semantic search, a cross-encoder reranker, and a small instruction-tuned language model to retrieve relevant information and generate answers.

## Features

* **Document ingestion:** Load `.txt` and text-based `.pdf` files from a local directory.
* **Chunking:** Split documents into smaller text segments with overlapping content.
* **Semantic search:** Generate multilingual embeddings using Sentence Transformers.
* **Reranking:** Reorder candidate chunks using a Cross-Encoder to improve relevance.
* **Answer generation:** Use `Qwen2.5-0.5B-Instruct` to answer questions based on retrieved context.
* **Source tracking:** Display the source documents used to construct an answer.
* **Index caching:** Save embeddings locally and reuse them when the indexed content and embedding model have not changed.
* **Basic automated tests:** Check numerical answers, mandatory approval conditions, and questions without supporting information.
* **Web interface:** Interact with the application through a Gradio interface.

## Project Structure

```text
mini_gpt/
├── app.py
├── rag.py
├── .gitignore
└── documenti/
    ├── policy_ferie.txt
    ├── policy_remoto.txt
    └── policy_rimborsi.txt
```

The file `rag_index.pt` is generated locally and is excluded from version control.

## Requirements

* Python
* PyTorch
* Transformers
* Sentence Transformers
* Gradio
* pypdf

## Installation

Clone the repository:

```bash
git clone https://github.com/ousmanembaye/mini-gpt-rag.git
cd mini-gpt-rag
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

The first run may download the embedding model, reranker, and language model. Internet access is needed for these downloads if the models are not already cached locally.

## Run the Application

Start the web interface:

```bash
python app.py
```

Open the local address shown in the terminal, normally:

http://127.0.0.1:7860

You can then submit questions about the documents stored in the `documenti/` directory.

## Example Questions

* How many vacation days does an employee receive?
* How many days per week is remote work allowed?
* Can an employee work remotely without manager approval?
* What is the annual training budget?

The last question is deliberately outside the example knowledge base. It can be used to test how the application handles missing information.

## How It Works

1. Documents are loaded from the `documenti/` directory.
2. Text is split into overlapping chunks.
3. An embedding model represents the chunks and user question as vectors.
4. Similarity search retrieves candidate chunks.
5. A Cross-Encoder reranks the candidates.
6. The selected context is provided to the language model.
7. The application returns an answer and the retrieved source names.

## Limitations

* PDF extraction works with text-based PDFs; scanned documents may require OCR.
* The language model is small and may generate incorrect or unsupported answers.
* Similarity and reranker thresholds are experimental and should be evaluated on a larger test set.
* The current automated tests cover only a small number of predefined cases.
* This project is an educational prototype, not a production-ready knowledge system.

## Future Improvements

* Add more robust evaluation and regression tests.
* Improve document retrieval and relevance filtering.
* Support additional document formats.
* Introduce configurable model and retrieval settings.
* Expand the project into a more complete AI assistant.

## License

No license has been specified yet.
