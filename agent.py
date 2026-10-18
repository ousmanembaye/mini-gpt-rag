import torch

from rag import (
    answer_query,
    documents,
    gen_model,
    gen_tokenizer,
    device
)


# TOOL 1: restituisce i nomi dei documenti disponibili
def list_policies() -> list[str]:

    return sorted({
        doc["source"]
        for doc in documents
    })


# TOOL 2: cerca una risposta tramite il RAG
def search_policies(question: str) -> dict:

    answer, sources, best_score = answer_query(question)

    return {
        "answer": answer,
        "sources": sources,
        "similarity": round(float(best_score), 4)
    }


# L'LLM sceglie quale strumento utilizzare
def choose_tool(user_message: str) -> str:

    messages = [
        {
            "role": "system",
            "content": """
Sei un router che deve scegliere uno strumento.

Puoi selezionare soltanto:

list_policies
search_policies

Regole:

- Se l'utente chiede quali documenti sono disponibili,
  scegli list_policies.

- Se l'utente fa una domanda sul contenuto dei documenti,
  scegli search_policies.

Esempi:

"Quali documenti hai?" -> list_policies

"Elencami le policy disponibili" -> list_policies

"Quanti giorni di ferie spettano?" -> search_policies

"Posso lavorare da remoto senza approvazione?"
-> search_policies

Rispondi esclusivamente con il nome dello strumento.
Non aggiungere spiegazioni.
"""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    prompt = gen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = gen_model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=gen_tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    new_tokens = output[0, input_length:]

    decision = gen_tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip().lower()

    print(f"\n[AGENT] Output del modello: {decision}")

    # Accettiamo la risposta del modello se contiene
    # un'etichetta valida all'inizio della risposta.
    first_line = decision.splitlines()[0].strip()

    if first_line.startswith("list_policies"):
        selected_tool = "list_policies"

    elif first_line.startswith("search_policies"):
        selected_tool = "search_policies"

    else:
        # Fallback: se il modello non rispetta il formato,
        # riconosciamo le richieste esplicite di elenco.
        normalized_question = user_message.casefold()

        list_phrases = [
            "quali documenti",
            "elenco dei documenti",
            "elenco documenti",
            "documenti disponibili",
            "lista dei documenti",
            "lista documenti",
            "mostra i documenti",
            "quali policy",
            "quali policies",
            "documenti hai a disposizione",
        ]

        if any(
            phrase in normalized_question
            for phrase in list_phrases
        ):
            selected_tool = "list_policies"
            print("[AGENT] Utilizzo il fallback per l'elenco.")

        else:
            selected_tool = "search_policies"
            print("[AGENT] Utilizzo il fallback per la ricerca.")

    print(f"[AGENT] Strumento selezionato: {selected_tool}")

    return selected_tool


# Esegue lo strumento selezionato e restituisce il risultato
def run_agent(user_message: str) -> str:

    selected_tool = choose_tool(user_message)

    if selected_tool == "list_policies":

        policies = list_policies()

        return (
            "Documenti disponibili:\n"
            + "\n".join(f"- {policy}" for policy in policies)
        )

    result = search_policies(user_message)

    sources = (
        ", ".join(result["sources"])
        if result["sources"]
        else "Nessuna fonte recuperata"
    )

    return (
        f"Risposta: {result['answer']}\n"
        f"Fonti: {sources}\n"
        f"Similarità: {result['similarity']}"
    )


# Avvio dell'agente
if __name__ == "__main__":

    print("\n===== MINI AI AGENT =====")
    print("Scrivi 'esci' per terminare.")

    while True:

        user_message = input("\nTu: ").strip()

        if user_message.lower() == "esci":
            print("Agente terminato.")
            break

        if not user_message:
            continue

        result = run_agent(user_message)

        print(f"\nAgente:\n{result}")