import unittest

from agent_routing import select_tool_from_output
from agent import run_agent, search_policies
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


if __name__ == "__main__":
    unittest.main(verbosity=2)