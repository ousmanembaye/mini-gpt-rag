import torch
from agent_routing import select_tool_from_output
from pathlib import Path
from tool_registry import TOOL_DESCRIPTIONS


# TOOL 1: restituisce i nomi dei documenti disponibili
def list_policies() -> list[str]:
    """Restituisce i nomi dei documenti disponibili senza caricare i modelli AI."""

    docs_dir = Path(__file__).resolve().parent / "documenti"

    if not docs_dir.is_dir():
        return []

    supported_extensions = {".txt", ".pdf"}

    return sorted(
        path.name
        for path in docs_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

def get_policy_details(policy_query: str) -> dict:
    """Restituisce il contenuto di una policy specifica senza caricare i modelli AI."""

    if not isinstance(policy_query, str) or not policy_query.strip():
        return {
            "ok": False,
            "policy": None,
            "content": "Indica quale policy vuoi consultare."
        }

    docs_dir = Path(__file__).resolve().parent / "documenti"

    if not docs_dir.is_dir():
        return {
            "ok": False,
            "policy": None,
            "content": "La cartella dei documenti non è disponibile."
        }

    # Individuiamo i documenti supportati
    paths = sorted(
        path
        for path in docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
    )

    # Sinonimi utili per identificare le policy di esempio
    aliases = {
        "ferie": ("ferie", "vacanza", "vacanze"),
        "remoto": (
            "remoto",
            "smart working",
            "telelavoro",
            "lavoro da casa"
        ),
        "rimborsi": (
            "rimborsi",
            "rimborso",
            "spese",
            "spesa",
            "ricevuta"
        )
    }

    query = policy_query.casefold()

    matches = []

    for path in paths:
        # Esempio: "policy_rimborsi" diventa "rimborsi"
        policy_name = path.stem.casefold()

        if policy_name.startswith("policy_"):
            policy_name = policy_name[len("policy_"):]

        policy_name = policy_name.replace("_", " ")

        keywords = aliases.get(
            policy_name,
            (policy_name,)
        )

        if any(keyword in query for keyword in keywords):
            matches.append(path)

    # Evitiamo di scegliere arbitrariamente se la richiesta è ambigua
    if len(matches) == 0:
        return {
            "ok": False,
            "policy": None,
            "content": (
                "Non ho identificato una policy specifica. "
                "Indica, per esempio, ferie, lavoro da remoto o rimborsi."
            )
        }

    if len(matches) > 1:
        return {
            "ok": False,
            "policy": None,
            "content": (
                "La richiesta corrisponde a più policy. "
                "Specifica quale documento vuoi consultare."
            )
        }

    path = matches[0]

    try:
        if path.suffix.lower() == ".txt":
            content = path.read_text(
                encoding="utf-8"
            ).strip()

        else:
            from pypdf import PdfReader

            reader = PdfReader(str(path))

            content = "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            ).strip()

    except Exception as error:
        print(f"[TOOL ERROR] get_policy_details: {error}")

        return {
            "ok": False,
            "policy": path.name,
            "content": "Non è stato possibile leggere il documento."
        }

    if not content:
        return {
            "ok": False,
            "policy": path.name,
            "content": (
                "Non è stato possibile estrarre testo dal documento."
            )
        }

    return {
        "ok": True,
        "policy": path.name,
        "content": content
    }


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
    from model_loader import gen_model, gen_tokenizer, device
    tool_descriptions = "\n".join(
        f"- {name}: {description}"
        for name, description in TOOL_DESCRIPTIONS.items()
    )

    messages = [
        {
            "role": "system",
            "content": f"""
Sei un router per un assistente aziendale.

Devi scegliere lo strumento più adatto alla richiesta
dell'utente tra quelli disponibili:

{tool_descriptions}

Regole:
- Se l'utente vuole sapere quali documenti esistono,
  scegli list_policies.
- Se vuole leggere il contenuto completo di una policy,
  scegli get_policy_details.
- Se pone una domanda e vuole una risposta basata sui documenti,
  scegli search_policies.

Rispondi esclusivamente con il nome esatto dello strumento.
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
            max_new_tokens=16,
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
def execute_tool(selected_tool: str, user_message: str) -> str:
    """Esegue lo strumento selezionato e formatta il risultato."""

    # Strumento 1: elenco dei documenti
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

        # Strumento 2: mostra il contenuto completo di una policy
    elif selected_tool == "get_policy_details":

        result = get_policy_details(user_message)

        if not result["ok"]:
            return result["content"]

        return (
            f"Contenuto della policy {result['policy']}:\n\n"
            f"{result['content']}"
        )

    # Strumento 2: ricerca nel RAG
    elif selected_tool == "search_policies":

        result = search_policies(user_message)

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

    # Gestione di uno strumento non riconosciuto
    return "Non sono riuscito a selezionare uno strumento valido."


def run_agent(user_message: str) -> str:
    """Gestisce la richiesta e coordina decisione ed esecuzione."""

    if not isinstance(user_message, str) or not user_message.strip():
        return "Inserisci una domanda per continuare."

    user_message = user_message.strip()

    # 1. Decidiamo quale strumento utilizzare
    selected_tool = choose_tool(user_message)

    # 2. Deleghiamo l'esecuzione a una funzione separata
    return execute_tool(selected_tool, user_message)

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