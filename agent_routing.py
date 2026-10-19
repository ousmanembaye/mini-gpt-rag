def select_tool_from_output(
    decision: str,
    user_message: str
) -> str:
    """
    Determina quale strumento utilizzare interpretando
    l'output del modello e, se necessario, applicando un fallback.
    """

    # Normalizziamo gli input
    if not isinstance(decision, str):
        decision = ""

    if not isinstance(user_message, str):
        user_message = ""

    decision = decision.strip().lower()
    normalized_question = user_message.casefold()

    # Esaminiamo la prima riga della risposta del modello
    first_line = (
        decision.splitlines()[0].strip()
        if decision
        else ""
    )

    # Caso 1: il modello ha restituito uno strumento valido
    if first_line.startswith("list_policies"):
        return "list_policies"

    if first_line.startswith("search_policies"):
        return "search_policies"

    # Caso 2: l'output del modello non è valido.
    # Applichiamo il fallback per le richieste di elenco.
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
        "documenti hai a disposizione"
    ]

    if any(
        phrase in normalized_question
        for phrase in list_phrases
    ):
        return "list_policies"

    # In assenza di un'indicazione chiara, usiamo la ricerca
    return "search_policies"