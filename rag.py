# http://127.0.0.1:7860 link per vedere il programma

# Carica un embedding model multilingue
import os
import torch
import hashlib
import json
from pypdf import PdfReader
from config import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    GENERATION_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
    RERANKER_THRESHOLD,
    MAX_NEW_TOKENS,
)

from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

# Cartella in cui si trova rag.py
BASE_DIR = Path(__file__).resolve().parent

# Cartella contenente i documenti
DOCS_DIR = BASE_DIR / "documenti"

documents = []

# Esaminiamo tutti i file della cartella
for path in sorted(DOCS_DIR.iterdir()):

    # Leggiamo i file di testo
    if path.suffix.lower() == ".txt":
        text = path.read_text(
            encoding="utf-8"
        ).strip()

    # Leggiamo i PDF
    elif path.suffix.lower() == ".pdf":

        reader = PdfReader(str(path))

        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)

        text = "\n".join(pages_text).strip()

    # Ignoriamo i formati non supportati
    else:
        continue

    # Controlliamo se è stato estratto del testo
    if not text:
        print(
            f"Attenzione: nessun testo estratto da {path.name}."
        )
        continue

    # Aggiungiamo il documento alla knowledge base
    documents.append({
        "source": path.name,
        "text": text
    })

# Verifichiamo che siano stati caricati documenti
if not documents:
    raise RuntimeError(
        f"Nessun documento leggibile trovato in {DOCS_DIR}"
    )


for doc in documents:
    print(f"- {doc['source']}")

# Verifichiamo che esista almeno un documento
if not documents:
    raise RuntimeError(
        f"Nessun documento .txt trovato in {DOCS_DIR}"
    )

print(f"Documenti caricati: {len(documents)}")

for doc in documents:
    print(f"- {doc['source']}")

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    words = text.split()

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Parametri di chunking non validi")

    chunks = []
    stride = chunk_size - overlap

    for start in range(0, len(words), stride):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks

all_chunks = []

for doc_index, doc in enumerate(documents):
    chunks = chunk_text(doc["text"])

    for chunk_index, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"{doc_index}_{chunk_index}",
            "source": doc["source"],
            "text": chunk
        })



# Carica il modello di embedding

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Percorso dell'indice salvato
index_path = BASE_DIR / "rag_index.pt"

# Creiamo una firma dei documenti e del modello utilizzato
index_data = {
    "chunks": all_chunks,
    "embedding_model": EMBEDDING_MODEL_NAME
}

index_signature = hashlib.sha256(
    json.dumps(
        index_data,
        ensure_ascii=False,
        sort_keys=True
    ).encode("utf-8")
).hexdigest()

# Controlliamo se esiste un indice precedente
saved = None

if index_path.exists():
    saved = torch.load(
        index_path,
        map_location="cpu",
        weights_only=False
    )

# Se documenti e modello non sono cambiati, riutilizziamo l'indice
if saved is not None and saved.get("signature") == index_signature:

    all_chunks = saved["chunks"]

    embeddings = saved["embeddings"].to(
        embedding_model.device
    )

    print("Indice esistente caricato dalla cache.")

# Altrimenti, ricostruiamo l'indice
else:

    print("Indice assente o non aggiornato: lo ricostruisco.")

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    torch.save(
        {
            "chunks": all_chunks,
            "embeddings": embeddings.detach().cpu(),
            "signature": index_signature
        },
        index_path
    )

    print("Nuovo indice salvato.")

print("Numero di chunk:", len(all_chunks))
print("Shape degli embeddings:", embeddings.shape)

# Estrae il testo di ogni chunk
texts = [chunk["text"] for chunk in all_chunks]

# Trasforma i testi in embedding
embeddings = embedding_model.encode(
    texts,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

import torch

# Importiamo il modello generativo condiviso
from model_loader import gen_model, gen_tokenizer, device

# Carichiamo il reranker una sola volta
reranker = CrossEncoder(
    RERANKER_MODEL_NAME,
    device=device
)

def answer_query(query, debug=False):
    # Parametri del retrieval
    similarity_threshold = SIMILARITY_THRESHOLD
    top_k = RETRIEVAL_TOP_K

    # 1. Trasformiamo la domanda in un embedding
    query_embedding = embedding_model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # 2. Calcoliamo la similarità con tutti i chunk
    similarities = embeddings @ query_embedding

    # 3. Recuperiamo i chunk con la similarità più alta
    scores, indices = torch.topk(
        similarities,
        k=min(top_k, len(all_chunks))
    )

        # Recuperiamo i chunk selezionati dall'embedding model
    candidate_chunks = [
        all_chunks[idx.item()]
        for idx in indices
    ]

    # Prepariamo le coppie domanda-documento
    pairs = [
        [query, chunk["text"]]
        for chunk in candidate_chunks
    ]

    # Valutiamo la pertinenza con il Cross-Encoder
    rerank_scores = reranker.predict(pairs)

    # Ordiniamo i documenti secondo il punteggio del reranker
    ranked_candidates = sorted(
        zip(rerank_scores, scores.tolist(), candidate_chunks),
        key=lambda item: float(item[0]),
        reverse=True
    )

    # Stampiamo i risultati

    # Stampiamo i risultati soltanto in modalità debug
    if debug:
        print("\n--- CONFRONTO DEI MODELLI ---")

        for rerank_score, embedding_score, chunk in ranked_candidates:
            print(f"\nEmbedding similarity: {embedding_score:.4f}")
            print(f"Reranker score: {float(rerank_score):.4f}")
            print(f"Fonte: {chunk['source']}")
            print(f"Testo: {chunk['text']}")

    if debug:
        print("\n--- DEBUG RETRIEVAL ---")

        for score, idx in zip(scores, indices):
            chunk = all_chunks[idx.item()]

            print(f"\nSimilarità: {score.item():.4f}")
            print(f"Fonte: {chunk['source']}")
            print(f"Testo: {chunk['text']}")

    best_score = scores[0].item()

    # 4. Se la domanda è troppo distante dai documenti, ci fermiamo
    if best_score < similarity_threshold:
        return (
            "Non ho informazioni sufficienti nei documenti per rispondere.",
            [],
            best_score
        )

# 5. Selezioniamo i chunk usando il reranker
    reranker_threshold = RERANKER_THRESHOLD

    relevant_chunks = [
        chunk
        for rerank_score, embedding_score, chunk in ranked_candidates
        if float(rerank_score) >= reranker_threshold
    ]

    # Se nessun chunk supera la soglia, non rispondiamo
    if not relevant_chunks:
        return (
            "Non ho informazioni sufficienti nei documenti per rispondere.",
            [],
            best_score
        )

    # 6. Costruiamo il contesto da fornire al modello
    context = "\n\n".join(
        f"Fonte: {chunk['source']}\n"
        f"Contenuto: {chunk['text']}"
        for chunk in relevant_chunks
    )

    # 7. Prepariamo il prompt
    messages = [
        {
            "role": "system",
            "content": """
Sei un assistente che risponde a domande sulle policy aziendali.

Il contesto fornito è la fonte delle informazioni.
Non contraddire mai una regola esplicitamente presente nel contesto.

Presta particolare attenzione alle condizioni obbligatorie:
- "previa approvazione" significa che l'approvazione è necessaria;
- "richiede autorizzazione" significa che l'autorizzazione è necessaria.

Se l'utente chiede se può fare qualcosa senza
un'autorizzazione che il documento dichiara obbligatoria,
rispondi di no e spiega brevemente la condizione.

Non inventare informazioni.

Usa esclusivamente le informazioni presenti nel contesto.

Quando la domanda richiede un numero, una data o una quantità,
riporta esattamente il valore presente nel contesto.
Non modificarlo e non sostituirlo con un valore diverso.

Esempio:
Contesto: Ogni dipendente ha diritto a 25 giorni di ferie all'anno.
Domanda: Quanti giorni di ferie spettano?
Risposta corretta: 25 giorni di ferie all'anno.

Se il contesto non contiene la risposta, dichiara
che non hai informazioni sufficienti.

Rispondi sempre in italiano, con una frase breve.
Non rispondere mai soltanto "Sì" o "No" quando la domanda
riguarda una regola aziendale: spiega brevemente la regola pertinente.

Quando una policy richiede un'approvazione, dichiaralo
esplicitamente nella risposta.

Esempio:
Domanda: Posso lavorare da remoto senza l'approvazione
del responsabile?
Risposta: No, è necessaria l'approvazione del responsabile.

Rispondi direttamente alla domanda, con una sola frase.

Non iniziare la risposta con titoli come "Oggetto:".

Non aggiungere condizioni o procedure che non siano
esplicitamente presenti nel contesto.

In particolare, non dedurre che una richiesta richieda
un'approvazione soltanto perché deve essere inviata
tramite un portale.

Dichiara che l'approvazione è obbligatoria soltanto
quando il contesto lo afferma esplicitamente.
"""
        },
        {
            "role": "user",
            "content": (
                f"Contesto:\n{context}\n\n"
                f"Domanda: {query}\n\n"
                "Rispondi con una frase breve."
            )
        }
    ]

    chat_text = gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 8. Tokenizziamo il prompt
    inputs = gen_tokenizer(
        chat_text,
        return_tensors="pt"
    ).to(device)

    # 9. Generiamo la risposta
    with torch.no_grad():
        output = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=gen_tokenizer.eos_token_id
        )

    # 10. Estraiamo solamente i token della risposta
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output[0, input_length:]

    answer = gen_tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    # 11. Recuperiamo le fonti utilizzate
    sources = list(
        dict.fromkeys(chunk["source"] for chunk in relevant_chunks)
    )

    return answer, sources, best_score

# ==============================
# TEST AUTOMATICI DEL RETRIEVAL
# ==============================
if __name__ == "__main__":
    # Qui vanno i test automatici
    # con tutte le righe indentate di 4 spazi

    test_cases = [
        (
            "Ferie",
            "Quanti giorni di ferie spettano a un dipendente?",
            "policy_ferie.txt"
        ),
        (
            "Lavoro da remoto",
            "Quanti giorni a settimana posso lavorare da remoto?",
            "policy_remoto.txt"
        ),
        (
            "Informazione assente",
            "Qual è il budget annuale per la formazione dei dipendenti?",
            None
        )
    ]

    test_threshold = 0.5
    test_min_chunk_similarity = 0.5
    test_top_k = 3

    passed_tests = 0

    print("\n===== TEST AUTOMATICI =====")

    for name, test_query, expected_source in test_cases:

        # 1. Generiamo l'embedding della domanda
        query_embedding = embedding_model.encode(
            test_query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # 2. Calcoliamo la similarità con tutti i chunk
        similarities = embeddings @ query_embedding

        # 3. Recuperiamo i migliori candidati
        scores, indices = torch.topk(
            similarities,
            k=min(test_top_k, len(all_chunks))
        )


        best_score = scores[0].item()

        # 4. Applichiamo il filtro ai singoli chunk
        retrieved_sources = [
            all_chunks[idx.item()]["source"]
            for score, idx in zip(scores, indices)
            if score.item() >= test_min_chunk_similarity
        ]

        # 5. Verifichiamo il risultato atteso
        if expected_source is not None:
            passed = (
                best_score >= test_threshold
                and expected_source in retrieved_sources
            )
        else:
            passed = (
                best_score < test_threshold
                and len(retrieved_sources) == 0
            )

        # 6. Stampiamo il risultato
        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {name}")
        print(f"Domanda: {test_query}")
        print(f"Similarità migliore: {best_score:.4f}")
        print(f"Fonti recuperate: {retrieved_sources}")

        if passed:
            passed_tests += 1

    print(
        f"\nRisultato finale: {passed_tests}/{len(test_cases)} "
        "test superati."
    )

    print("=============================\n")

    generation_tests = [
        {
            "name": "Numero di giorni di ferie",
            "query": "Quanti giorni di ferie spettano a un dipendente?",
            "type": "number"
        },
        {
            "name": "Approvazione obbligatoria",
            "query": (
                "Posso lavorare da remoto senza "
                "l'approvazione del responsabile?"
            ),
            "type": "approval"
        },
        {
            "name": "Informazione assente",
            "query": (
                "Qual è il budget annuale per la formazione "
                "dei dipendenti?"
            ),
            "type": "missing"
        }
    ]

    passed_tests = 0

    print("\n===== TEST AUTOMATICI DELLA GENERAZIONE =====")

    for test in generation_tests:
        answer, sources, best_score = answer_query(test["query"])

        answer_lower = answer.lower().strip()

        # Test 1: la risposta deve contenere il numero corretto
        if test["type"] == "number":
            passed = (
            answer_lower.startswith(("25", "sono 25"))
            and "giorni di ferie" in answer_lower
            and "approvazione" not in answer_lower
    )

        # Test 2: la risposta deve negare il permesso
        # e indicare la condizione relativa all'approvazione
        elif test["type"] == "approval":
            starts_with_denial = answer_lower.startswith(("no", "non"))
            mentions_condition = (
                "approvazione" in answer_lower
                or "responsabile" in answer_lower
            )

            passed = starts_with_denial and mentions_condition

        # Test 3: se l'informazione manca, non deve inventarla
        elif test["type"] == "missing":
            passed = (
                "non ho informazioni sufficienti" in answer_lower
            )

        print(f"\n{'[PASS]' if passed else '[FAIL]'} {test['name']}")
        print(f"Domanda: {test['query']}")
        print(f"Risposta: {answer}")
        print(f"Similarità migliore: {best_score:.4f}")
        print(f"Fonti recuperate: {sources}")

        if passed:
            passed_tests += 1

    print(
        f"\nRisultato finale: {passed_tests}/{len(generation_tests)} "
        "test superati."
    )

    print("=============================================\n")

    while True:
        query = input("\nFai una domanda (oppure 'esci'): ").strip()

        if query.lower() == "esci":
            print("Programma terminato.")
            break

        if not query:
            continue

        answer, sources, best_score = answer_query(query)

        print("\nRisposta:", answer)

        if sources:
            print("Fonti:", ", ".join(sources))
        else:
            print("Fonti: nessuna")

        print(f"Similarità migliore: {best_score:.4f}")