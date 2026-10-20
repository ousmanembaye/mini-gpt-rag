def select_tool_from_output(
    decision: str,
    user_message: str
) -> str:
    """
    Seleziona lo strumento considerando prima
    l'intenzione esplicita dell'utente.
    """

    if not isinstance(decision, str):
        decision = ""

    if not isinstance(user_message, str):
        user_message = ""

    decision = decision.strip().lower()
    normalized_question = user_message.casefold()

    # 1. Riconosciamo le richieste di elenco
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
        return "list_policies"

    # 2. Riconosciamo le richieste di lettura completa
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
    ]

    if any(
        phrase in normalized_question
        for phrase in detail_phrases
    ):
        return "get_policy_details"

    # 3. Per le normali domande sui contenuti,
    # accettiamo search_policies se il modello
    # lo ha selezionato.
    first_line = (
        decision.splitlines()[0].strip()
        if decision
        else ""
    )

    if first_line.startswith("search_policies"):
        return "search_policies"

    # 4. Se il modello propone un altro strumento
    # senza una corrispondente intenzione esplicita,
    # usiamo la ricerca nel RAG come comportamento predefinito.
    return "search_policies"