import unittest

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


if __name__ == "__main__":
    unittest.main(verbosity=2)