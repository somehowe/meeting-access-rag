from io import BytesIO, StringIO
import json
import unittest
from unittest.mock import patch

from meeting_rag import DATA, Index, Passage, ask_ollama, load_passages, main, parse_claims


class RetrievalTests(unittest.TestCase):
    def test_bm25_values_rare_match_over_common_term(self):
        passages = [
            Passage("a", "", "meeting agenda rooms lunch plans"),
            Passage("b", "", "meeting interpreter interpreter review plans"),
            Passage("c", "", "meeting slides chairs order plans"),
        ]
        hits = Index(passages).search("meeting interpreter", k=1)
        self.assertEqual(hits[0].passage.id, "b")

    def test_sample_question_retrieves_both_sources(self):
        question = "How far ahead do I book the interpreter and share the accessible slide deck?"
        index = Index(load_passages(DATA / "docs.json"))
        sources = {hit.passage.id for hit in index.search(question, k=2)}
        self.assertEqual(sources, {"interpreter-request", "materials"})

    def test_empty_retrieval_skips_the_model(self):
        with (
            patch("sys.argv", ["meeting_rag.py", "quasar redshift", "--model", "fake"]),
            patch("sys.stdout", new_callable=StringIO) as stdout,
            patch("meeting_rag.ask_ollama") as ask,
        ):
            self.assertEqual(main(), 0)
        ask.assert_not_called()
        self.assertIn("enough evidence", stdout.getvalue())


class GenerationTests(unittest.TestCase):
    def test_requests_a_structured_answer_with_retrieved_context(self):
        index = Index([Passage("source", "Interpreter", "Book the interpreter early.")])
        hits = index.search("book interpreter")
        claims = [{"text": "Book the interpreter early.", "sources": ["source"]}]
        response = {"message": {"content": json.dumps({"claims": claims})}}
        with patch("meeting_rag.urlopen", return_value=BytesIO(json.dumps(response).encode())) as send:
            self.assertEqual(ask_ollama("When do I book?", hits, "test-model"), claims)

        request = send.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        payload = json.loads(request.data)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(json.loads(payload["messages"][1]["content"])["passages"], [
            {"id": "source", "title": "Interpreter", "text": "Book the interpreter early."},
        ])

    def test_rejects_claim_with_unretrieved_citation(self):
        bad = '{"claims": [{"text": "Claim", "sources": ["unknown"]}]}'
        with self.assertRaisesRegex(ValueError, "valid source"):
            parse_claims(bad, {"retrieved"})


if __name__ == "__main__":
    unittest.main()
