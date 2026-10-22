TOOL_DESCRIPTIONS = {
    "list_policies": (
        "Elenca i documenti e le policy disponibili. "
        "Usalo quando l'utente vuole sapere quali file esistono."
    ),

    "search_policies": (
        "Risponde a domande sul contenuto delle policy utilizzando "
        "il sistema RAG. Usalo per domande su ferie, rimborsi, "
        "lavoro da remoto e altre informazioni contenute nei documenti."
    ),

    "get_policy_details": (
        "Legge e restituisce il contenuto completo di una specifica "
        "policy. Usalo quando l'utente vuole leggere o visualizzare "
        "un intero documento, anziché ricevere una risposta sintetica."
    ),
}


def get_tool_names() -> tuple[str, ...]:
    """Restituisce i nomi degli strumenti registrati."""

    return tuple(TOOL_DESCRIPTIONS.keys())