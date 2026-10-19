import torch
from agent_routing import select_tool_from_output


# TOOL 1: restituisce i nomi dei documenti disponibili
def list_policies() -> list[str]:
    """Restituisce i nomi dei documenti disponibili."""

    from rag import documents

    return sorted({
        doc["source"]
        for doc in documents
    })


# TOOL 2: cerca una risposta tramite il RAG
def search_policies(question: str) -> dict:
    """Cerca una risposta nel RAG e gestisce gli errori."""

    # 1. Verifichiamo la domanda
    if not isinstance(question, str) or not question.strip():
        return {
            "ok": False,
            "answer": "Inserisci una domanda valida.",
            "sources": [],
            "similarity": None
        }

    # 2. Eseguiamo il RAG e gestiamo eventuali errori
    try:
        from rag import answer_query
        answer, sources, best_score = answer_query(question.strip())

    except Exception as error:
        print(f"\n[TOOL ERROR] search_policies: {error}")

        return {
            "ok": False,
            "answer": (
                "Si è verificato un errore durante la ricerca "
                "nei documenti."
            ),
            "sources": [],
            "similarity": None
        }

    # 3. Restituiamo il risultato
    return {
        "ok": True,
        "answer": answer,
        "sources": sources,
        "similarity": round(float(best_score), 4)
    }


# L'LLM sceglie quale strumento utilizzare
def choose_tool(user_message: str) -> str:
    from rag import gen_model, gen_tokenizer, device

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

    # Interpretiamo la risposta del modello usando il router
    selected_tool = select_tool_from_output(
        decision,
        user_message
    )

    print(f"[AGENT] Strumento selezionato: {selected_tool}")

    return selected_tool


# Esegue lo strumento selezionato e restituisce il risultato
def run_agent(user_message: str) -> str:
    """Gestisce la richiesta dell'utente ed esegue lo strumento."""

    # 1. Controlliamo l'input prima di chiamare il modello
    if not isinstance(user_message, str) or not user_message.strip():
        return "Inserisci una domanda per continuare."

    user_message = user_message.strip()

    # 2. Il modello seleziona lo strumento
    selected_tool = choose_tool(user_message)

    # 3. Eseguiamo lo strumento selezionato
    if selected_tool == "list_policies":

        try:
            policies = list_policies()

        except Exception as error:
            print(f"\n[TOOL ERROR] list_policies: {error}")
            return "Non è stato possibile recuperare l'elenco dei documenti."

        if not policies:
            return "Non sono disponibili documenti."

        return (
            "Documenti disponibili:\n"
            + "\n".join(f"- {policy}" for policy in policies)
        )

    elif selected_tool == "search_policies":

        result = search_policies(user_message)

        # Gestiamo il fallimento dello strumento
        if not result["ok"]:
            return result["answer"]

        sources = (
            ", ".join(result["sources"])
            if result["sources"]
            else "Nessuna fonte recuperata"
        )

        similarity = result["similarity"]

        similarity_text = (
            f"{similarity:.4f}"
            if similarity is not None
            else "N/A"
        )

        return (
            f"Risposta: {result['answer']}\n"
            f"Fonti: {sources}\n"
            f"Similarità: {similarity_text}"
        )

    # 4. Gestiamo eventuali strumenti non riconosciuti
    return "Non sono riuscito a selezionare uno strumento valido."


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