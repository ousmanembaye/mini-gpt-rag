import unittest

from agent_routing import select_tool_from_output
from agent import (
    run_agent,
    search_policies,
    get_policy_details
)
from unittest.mock import patch


class AgentInputTests(unittest.TestCase):

    def test_empty_agent_input(self):
        """Un input vuoto non deve richiedere l'intervento del modello."""

        result = run_agent("   ")

        self.assertEqual(
            result,
            "Inserisci una domanda per continuare."
        )

    def test_empty_search_input(self):
        """Una domanda vuota non deve avviare il RAG."""

        result = search_policies(" ")

        self.assertFalse(result["ok"])
        self.assertEqual(result["sources"], [])
        self.assertIsNone(result["similarity"])
        self.assertIn(
            "domanda valida",
            result["answer"].lower()
        )
    def test_valid_list_tool_output(self):
        result = select_tool_from_output(
            "list_policies",
            "Quali documenti hai a disposizione?"
        )

        self.assertEqual(result, "list_policies")

    def test_invalid_output_fallback(self):
        result = select_tool_from_output(
            "Non ho informazioni sufficienti.",
            "Quali documenti hai a disposizione?"
        )

        self.assertEqual(result, "list_policies")

    def test_valid_search_tool_output(self):
        result = select_tool_from_output(
            "search_policies",
            "Quanti giorni di ferie spettano?"
        )

        self.assertEqual(result, "search_policies")
    def test_run_agent_executes_list_tool(self):
        question = "Quali documenti hai a disposizione?"

        policies = [
            "policy_ferie.txt",
            "policy_remoto.txt",
            "policy_rimborsi.txt"
        ]

        with patch(
            "agent.choose_tool",
            return_value="list_policies"
        ), patch(
            "agent.list_policies",
            return_value=policies
        ) as mock_list, patch(
            "agent.search_policies"
        ) as mock_search:

            result = run_agent(question)

        self.assertIn("policy_ferie.txt", result)
        self.assertIn("policy_remoto.txt", result)

        mock_list.assert_called_once_with()
        mock_search.assert_not_called()
    def test_run_agent_executes_search_tool(self):
        question = "Quanti giorni di ferie spettano?"

        fake_result = {
            "ok": True,
            "answer": "25 giorni di ferie all'anno.",
            "sources": ["policy_ferie.txt"],
            "similarity": 0.7051
        }

        with patch(
            "agent.choose_tool",
            return_value="search_policies"
        ), patch(
            "agent.search_policies",
            return_value=fake_result
        ) as mock_search, patch(
            "agent.list_policies"
        ) as mock_list:

            result = run_agent(question)

        self.assertIn("25 giorni di ferie", result)
        self.assertIn("policy_ferie.txt", result)

        mock_search.assert_called_once_with(question)
        mock_list.assert_not_called()
    def test_run_agent_handles_tool_error(self):
        question = "Quanti giorni di ferie spettano?"

        fake_error = {
            "ok": False,
            "answer": (
                "Si è verificato un errore durante la ricerca "
                "nei documenti."
            ),
            "sources": [],
            "similarity": None
        }

        with patch(
            "agent.choose_tool",
            return_value="search_policies"
        ), patch(
            "agent.search_policies",
            return_value=fake_error
        ):

            result = run_agent(question)

        self.assertEqual(result, fake_error["answer"])
    def test_detail_request_overrides_search_decision(self):
        result = select_tool_from_output(
            "search_policies",
            "Mostrami il contenuto completo della policy sui rimborsi."
        )

        self.assertEqual(result, "get_policy_details")

    def test_normal_question_overrides_detail_decision(self):
        result = select_tool_from_output(
            "get_policy_details",
            "Quanti giorni di ferie spettano a un dipendente?"
        )

        self.assertEqual(result, "search_policies")
    def test_list_request_overrides_detail_decision(self):
        result = select_tool_from_output(
            "get_policy_details",
            "Quali documenti hai a disposizione?"
        )

        self.assertEqual(result, "list_policies")
    def test_run_agent_executes_policy_details_tool(self):
        question = (
            "Mostrami il contenuto completo "
            "della policy sui rimborsi."
        )

        fake_result = {
            "ok": True,
            "policy": "policy_rimborsi.txt",
            "content": "Testo di esempio della policy sui rimborsi."
        }

        with patch(
            "agent.choose_tool",
            return_value="get_policy_details"
        ), patch(
            "agent.get_policy_details",
            return_value=fake_result
        ) as mock_details, patch(
            "agent.search_policies"
        ) as mock_search:

            result = run_agent(question)

        self.assertIn("policy_rimborsi.txt", result)
        self.assertIn("Testo di esempio", result)

        mock_details.assert_called_once_with(question)
        mock_search.assert_not_called()
    def test_policy_details_returns_correct_document(self):
        result = get_policy_details(
            "Mostrami il contenuto completo della policy sui rimborsi"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["policy"],
            "policy_rimborsi.txt"
        )
        self.assertIn("30 giorni", result["content"])
        self.assertIn("ricevuta", result["content"].lower())
    def test_policy_details_empty_input(self):
        result = get_policy_details("   ")

        self.assertFalse(result["ok"])
        self.assertIsNone(result["policy"])
        self.assertIn(
            "indica quale policy",
            result["content"].lower()
        )
    def test_policy_details_unknown_document(self):
        result = get_policy_details(
            "Mostrami la policy sulla formazione"
        )

        self.assertFalse(result["ok"])
        self.assertIsNone(result["policy"])
        self.assertIn(
            "non ho identificato",
            result["content"].lower()
        )
    def test_policy_details_ambiguous_request(self):
        result = get_policy_details(
            "Mostrami le policy su ferie e rimborsi"
        )

        self.assertFalse(result["ok"])
        self.assertIsNone(result["policy"])
        self.assertIn(
            "più policy",
            result["content"].lower()
        )
    def test_alternative_list_request(self):
        result = select_tool_from_output(
            "list_policies",
            "Puoi indicarmi quali file sono presenti?"
        )

        self.assertEqual(result, "list_policies")

    def test_alternative_detail_request(self):
        result = select_tool_from_output(
            "search_policies",
            "Vorrei consultare il documento sui rimborsi."
        )

        self.assertEqual(result, "get_policy_details")
    def test_routing_paraphrases(self):
        test_cases = [
            # Richieste di elenco
            (
                "Elenco alternativo",
                "Elenca tutti i documenti presenti",
                "get_policy_details",
                "list_policies"
            ),
            (
                "Ricerca dei file disponibili",
                "Quali file sono disponibili?",
                "search_policies",
                "list_policies"
            ),
            (
                "Elenco delle policy",
                "Fammi vedere l'elenco delle policy aziendali",
                "search_policies",
                "list_policies"
            ),

            # Richieste di lettura completa
            (
                "Consultazione dei rimborsi",
                "Vorrei consultare la policy sui rimborsi.",
                "search_policies",
                "get_policy_details"
            ),
            (
                "Lettura della policy ferie",
                "Fammi vedere la policy delle ferie",
                "search_policies",
                "get_policy_details"
            ),
            (
                "Testo completo della policy",
                "Puoi mostrarmi il testo completo della policy per il lavoro da remoto?",
                "search_policies",
                "get_policy_details"
            ),

            # Domande che richiedono il RAG
            (
                "Giorni di ferie",
                "Quanti giorni di ferie mi spettano?",
                "get_policy_details",
                "search_policies"
            ),
            (
                "Procedura di rimborso",
                "Come richiedo un rimborso?",
                "get_policy_details",
                "search_policies"
            ),
            (
                "Scadenza del rimborso",
                "Quando devo inviare la richiesta di rimborso?",
                "get_policy_details",
                "search_policies"
            ),
        ]

        for name, question, model_output, expected_tool in test_cases:
            with self.subTest(caso=name):
                result = select_tool_from_output(
                    model_output,
                    question
                )

                self.assertEqual(result, expected_tool)


if __name__ == "__main__":
    unittest.main(verbosity=2)