from pathlib import Path


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