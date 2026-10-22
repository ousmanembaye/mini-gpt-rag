import re


def normalize_text(text: str) -> str:
    """Normalizza il testo per rendere più robusti i controlli."""

    if not isinstance(text, str):
        return ""

    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def select_tool_from_output(
    decision: str,
    user_message: str
) -> str:
    """
    Seleziona uno strumento combinando regole esplicite
    e decisione del modello.
    """

    question = normalize_text(user_message)

    # Conserviamo la prima riga della decisione del modello
    if isinstance(decision, str):
        first_line = decision.strip().splitlines()
        model_decision = (
            normalize_text(first_line[0])
            if first_line
            else ""
        )
    else:
        model_decision = ""

    # 1. Riconosciamo le richieste di elenco
    list_phrases = [
        "quali documenti",
        "quali sono i documenti",
        "elenco dei documenti",
        "elenco documenti",
        "documenti disponibili",
        "documenti a disposizione",
        "lista dei documenti",
        "lista documenti",
        "quali policy",
        "quali policies",
        "quali sono le policy",
        "documenti hai a disposizione",
        "che documenti",
        "quali file",
        "che file",
        "elenco delle policy",
        "lista delle policy",
        "policy disponibili",
    ]

    if any(phrase in question for phrase in list_phrases):
        return "list_policies"

    # Riconosciamo anche alcune formulazioni alternative
    words = set(question.split())

    has_document = bool(
        words.intersection({
            "documento",
            "documenti",
            "file",
            "policy",
            "policies",
        })
    )

    has_list_action = bool(
        words.intersection({
            "elenca",
            "elencami",
            "elenco",
            "lista",
        })
    )

    has_availability = bool(
        words.intersection({
            "disponibile",
            "disponibili",
            "presenti",
            "esistono",
        })
    )

    if has_document and (
        has_list_action
        or has_availability
    ):
        return "list_policies"

    # 2. Riconosciamo le richieste di lettura di una policy
    detail_phrases = [
        "contenuto completo",
        "mostrami il contenuto",
        "mostra il contenuto",
        "fammi leggere",
        "leggi la policy",
        "leggi il documento",
        "mostra il testo completo",
        "testo completo della policy",
        "contenuto della policy",
        "mostrami la policy",
        "mostra la policy",
        "fammi vedere la policy",
        "apri la policy",
        "visualizza il documento",
        "visualizza la policy",
        "mostra il documento",
        "mostrami il documento",
        "vorrei leggere la policy",
        "vorrei consultare la policy",
        "consultare il documento",
        "consultare la policy",
        "mi fai vedere la policy",
        "fammi vedere il documento",
        "puoi mostrarmi la policy",
    ]

    if any(phrase in question for phrase in detail_phrases):
        return "get_policy_details"

    # 3. Una domanda informativa richiede normalmente il RAG.
    # Questo impedisce, per esempio, che una domanda sui giorni
    # di ferie venga trattata come una richiesta di lettura completa.
    search_starts = (
        "quanti ",
        "quante ",
        "quanta ",
        "quanto ",
        "posso ",
        "come ",
        "quando ",
        "perché ",
        "perche ",
        "qual e ",
        "è consentito ",
        "è necessario ",
        "si può ",
        "si puo ",
        "è possibile ",
        "e possibile ",
        "serve ",
        "occorre ",
        "entro quando ",
    )

    if question.startswith(search_starts):
        return "search_policies"

    # 4. Se le regole precedenti non hanno chiarito l'intenzione,
    # accettiamo una decisione valida del modello.
    valid_tools = {
        "list_policies",
        "search_policies",
        "get_policy_details",
    }

    if model_decision in valid_tools:
        return model_decision

    # 5. Fallback: ricerca nel RAG
    return "search_policies"