import unittest

from agent_routing import select_tool_from_output
from agent import run_agent, search_policies


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


if __name__ == "__main__":
    unittest.main(verbosity=2)