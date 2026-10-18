from agent import choose_tool


test_cases = [
    (
        "Elenco dei documenti",
        "Quali documenti hai a disposizione?",
        "list_policies"
    ),
    (
        "Domanda sulle ferie",
        "Quanti giorni di ferie spettano a un dipendente?",
        "search_policies"
    ),
    (
        "Approvazione per il lavoro da remoto",
        "Posso lavorare da remoto senza l'approvazione del responsabile?",
        "search_policies"
    ),
    (
        "Informazione assente",
        "Qual è il budget annuale per la formazione dei dipendenti?",
        "search_policies"
    )
]


passed = 0

print("\n===== TEST AUTOMATICI DELL'AGENTE =====")

for name, question, expected_tool in test_cases:

    actual_tool = choose_tool(question)

    success = actual_tool == expected_tool

    print(f"\n[{'PASS' if success else 'FAIL'}] {name}")
    print(f"Domanda: {question}")
    print(f"Strumento atteso: {expected_tool}")
    print(f"Strumento selezionato: {actual_tool}")

    if success:
        passed += 1

print(f"\nRisultato: {passed}/{len(test_cases)} test superati.")
print("========================================")