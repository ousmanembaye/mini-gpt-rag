from agent import (
    choose_tool,
    list_policies,
    search_policies,
    run_agent
)


def run_test(name, test_function):
    """Esegue un test e registra il risultato."""

    try:
        passed = bool(test_function())
    except Exception as error:
        passed = False
        print(f"\nErrore durante il test: {error}")

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {name}")

    return passed


# ==================================================
# TEST 1-4: SELEZIONE DEGLI STRUMENTI
# ==================================================

def test_list_selection():
    return (
        choose_tool("Quali documenti hai a disposizione?")
        == "list_policies"
    )


def test_vacation_selection():
    return (
        choose_tool("Quanti giorni di ferie spettano a un dipendente?")
        == "search_policies"
    )


def test_remote_selection():
    return (
        choose_tool(
            "Posso lavorare da remoto senza l'approvazione del responsabile?"
        )
        == "search_policies"
    )


def test_missing_info_selection():
    return (
        choose_tool(
            "Qual è il budget annuale per la formazione dei dipendenti?"
        )
        == "search_policies"
    )


# ==================================================
# TEST 5: ESECUZIONE DELLO STRUMENTO DI ELENCO
# ==================================================

def test_list_result():
    policies = list_policies()

    expected = {
        "policy_ferie.txt",
        "policy_remoto.txt",
        "policy_rimborsi.txt"
    }

    return expected.issubset(set(policies))


# ==================================================
# TEST 6: RISULTATO DELLA RICERCA SULLE FERIE
# ==================================================

def test_vacation_result():
    result = search_policies(
        "Quanti giorni di ferie spettano a un dipendente?"
    )

    answer = result["answer"].lower()

    return (
        "25" in answer
        and "giorni di ferie" in answer
        and "policy_ferie.txt" in result["sources"]
    )


# ==================================================
# TEST 7: RISPETTO DELL'APPROVAZIONE OBBLIGATORIA
# ==================================================

def test_approval_result():
    result = search_policies(
        "Posso lavorare da remoto senza l'approvazione del responsabile?"
    )

    answer = result["answer"].strip().lower()

    return (
        answer.startswith(("no", "non"))
        and (
            "approvazione" in answer
            or "responsabile" in answer
        )
        and "policy_remoto.txt" in result["sources"]
    )


# ==================================================
# TEST 8: GESTIONE DELLE INFORMAZIONI MANCANTI
# ==================================================

def test_missing_information_result():
    result = search_policies(
        "Qual è il budget annuale per la formazione dei dipendenti?"
    )

    return (
        "non ho informazioni sufficienti" in result["answer"].lower()
        and len(result["sources"]) == 0
    )

# ==================================================
# TEST 9: INPUT VUOTO DELL'AGENTE
# ==================================================

def test_empty_agent_input():
    result = run_agent("   ")

    return result == "Inserisci una domanda per continuare."


# ==================================================
# TEST 10: INPUT NON VALIDO PER IL RAG
# ==================================================

def test_empty_search_input():
    result = search_policies(" ")

    return (
        result["ok"] is False
        and result["sources"] == []
        and result["similarity"] is None
        and "domanda valida" in result["answer"].lower()
    )
def test_policy_details_result():
    """Verifica che l'agente restituisca il contenuto della policy sui rimborsi."""

    result = run_agent(
        "Mostrami il contenuto completo della policy sui rimborsi."
    )

    return (
        "policy_rimborsi.txt" in result
        and "30 giorni" in result.lower()
        and "ricevuta" in result.lower()
    )

# ==================================================
# ESECUZIONE DI TUTTI I TEST
# ==================================================

tests = [
    ("Selezione dello strumento per l'elenco", test_list_selection),
    ("Selezione dello strumento per le ferie", test_vacation_selection),
    ("Selezione dello strumento per il lavoro da remoto", test_remote_selection),
    ("Selezione dello strumento per informazioni assenti", test_missing_info_selection),
    ("Risultato dell'elenco dei documenti", test_list_result),
    ("Risposta sulle ferie e fonte corretta", test_vacation_result),
    ("Rispetto dell'approvazione obbligatoria", test_approval_result),
    ("Gestione delle informazioni mancanti", test_missing_information_result),
    ("Gestione dell'input vuoto nell'agente", test_empty_agent_input),
    ("Gestione dell'input non valido nel RAG", test_empty_search_input),
    ("Lettura completa della policy sui rimborsi", test_policy_details_result),
]

passed = 0

print("\n===== TEST AUTOMATICI DELL'AI AGENT =====")

for name, test_function in tests:
    if run_test(name, test_function):
        passed += 1

print(f"\nRisultato finale: {passed}/{len(tests)} test superati.")
print("=========================================")