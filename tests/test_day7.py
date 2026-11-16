"""Offline tests for src/day7.py (no Day 6 data, no network; the model client is fake)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import day7  # noqa: E402
from baseline import hit  # noqa: E402  (starter, put on sys.path by day7)

SRC = day7.SOURCE_ID


def chunk(cid, text, page=1, page_end=None, book=1, chapter=1, index=0):
    return {"id": cid, "text": text, "source_id": SRC, "source": "hp.pdf", "page": page,
            "metadata": {"page_end": page_end or page, "book_no": book, "chapter_no": chapter,
                         "book_title": "B", "chapter_title": "T", "chunk_index": index}}


CHUNKS = [
    chunk("C-b1-c01-0000", "Mr. Dursley was the director of a firm called Grunnings, which made drills.", 12, index=0),
    chunk("C-b1-c07-0027", "“Password?” she said.", 121, chapter=7, index=27),
    chunk("C-b1-c07-0028", "“Caput Draconis,” said Percy, and the portrait swung forward.", 121, 122, chapter=7, index=28),
    chunk("C-b1-c14-0008", "Hagrid won the dragon egg in a game of cards.", 209, chapter=14, index=8),
    chunk("C-b3-c11-0037", "Harry got a Firebolt; it had to be checked for jinxes.", 766, book=3, chapter=11, index=37),
]
BY_ID = {c["id"]: c for c in CHUNKS}


class TextAndBenchmarkTests(unittest.TestCase):
    def test_verbatim_span_keeps_original_quotes(self):
        span = day7.verbatim_span('"Caput Draconis," said Percy', CHUNKS[2]["text"])
        self.assertEqual(span, "“Caput Draconis,” said Percy")
        self.assertIsNone(day7.verbatim_span("absent text", CHUNKS[2]["text"]))

    def test_tokenize(self):
        self.assertEqual(day7.tokenize("What is Hagrid's dragon?"), ["hagrid", "dragon"])

    def test_starter_hit_accepts_page_inside_chunk_span(self):
        gold = [{"source_id": SRC, "page": 122, "evidence": "portrait swung forward"}]
        self.assertTrue(hit(CHUNKS[2], gold))
        self.assertFalse(hit(CHUNKS[2], [{**gold[0], "page": 123}]))
        self.assertFalse(hit(CHUNKS[1], gold))

    def test_locate_copies_verbatim_and_rejects_cross_chunk_quotes(self):
        ev, how = day7.locate('"Caput Draconis," said Percy', 121, CHUNKS)
        self.assertEqual(ev, "“Caput Draconis,” said Percy")
        self.assertIn("normalized", how)
        self.assertIsNone(day7.locate("“Password?” she said. “Caput Draconis,”", 121, CHUNKS))


class RetrieverTests(unittest.TestCase):
    def test_bm25_rare_name_and_filter(self):
        bm25 = day7.BM25(CHUNKS)
        self.assertEqual(bm25.search("firm Grunnings")[0]["id"], "C-b1-c01-0000")
        self.assertEqual([r["id"] for r in bm25.search("Firebolt jinxes", filters={"book_no": 3})], ["C-b3-c11-0037"])
        self.assertEqual(bm25.search("Firebolt", filters={"book_no": 1}), [])
        with self.assertRaises(ValueError):
            bm25.search("x", filters={"page": 1})

    def test_tfidf_wrapper_applies_filter(self):
        tfidf = day7.TfidfRetriever(CHUNKS)
        self.assertEqual([r["id"] for r in tfidf.search("Firebolt", filters={"book_no": 3})], ["C-b3-c11-0037"])
        self.assertEqual(tfidf.search("Firebolt", filters={"book_no": 1}), [])

    def test_rrf_rewards_agreement(self):
        class Fixed:
            def __init__(self, name, ids): self.name, self.ids = name, ids
            def search(self, q, *, filters=None, top_k=3):
                return [{**BY_ID[i], "rank": r, "score": 1.0} for r, i in enumerate(self.ids[:top_k], 1)]
        ids = [c["id"] for c in CHUNKS]
        fused = day7.HybridRRF([Fixed("a", ids[0:3]), Fixed("b", [ids[2], ids[3], ids[4]])]).search("q")
        # ids[2] is found by both and beats ids[0]; ids[1] and ids[3] tie at 1/62 -> ID order
        self.assertEqual([r["id"] for r in fused], [ids[2], ids[0], ids[1]])
        self.assertEqual(fused[0]["component_ranks"], {"a": 3, "b": 1})

    def test_rerank_reorders(self):
        class Model:
            def predict(self, pairs): return [len(p[1]) for p in pairs]
        out = day7.CrossEncoderRerank(day7.BM25(CHUNKS), Model()).search("Firebolt jinxes Grunnings", top_k=2)
        self.assertEqual(out[0]["id"], "C-b1-c01-0000")
        self.assertIn("base_rank", out[0])

    def test_neighbours_keep_hits_first_and_fit_run_agent_limit(self):
        nb = day7.neighbour_index(CHUNKS)
        self.assertEqual(nb["C-b1-c07-0028"], ("C-b1-c07-0027", None))
        rows = day7.with_neighbours([CHUNKS[2], CHUNKS[0]], BY_ID, nb)
        self.assertEqual([r["id"] for r in rows], ["C-b1-c07-0028", "C-b1-c01-0000", "C-b1-c07-0027"])

        class Base:
            name = "base"
            def search(self, q, *, filters=None, top_k=3): return [{**CHUNKS[2], "rank": 1, "score": 1.0}]
        r = day7.AgentRetriever(Base(), 3, True, BY_ID, nb)
        self.assertEqual(r.max_per_search, 9)
        self.assertEqual([c["id"] for c in r.search("q", filters=None, top_k=9)], ["C-b1-c07-0028", "C-b1-c07-0027"])


def response(*blocks, stop="tool_use"):
    usage = SimpleNamespace(input_tokens=100, output_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=list(blocks), stop_reason=stop, usage=usage, model="claude-opus-5-5")


def tool(name, **inp):
    return SimpleNamespace(type="tool_use", id=f"id-{name}", name=name, input=inp)


class FakeClient:
    def __init__(self, responses):
        self.responses, self.prompts = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        self.kwargs = kw
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AgentTests(unittest.TestCase):
    """Claude controller inside the starter run_agent loop."""

    def run_q(self, responses, question="q", answerable=True, flt=None, max_searches=3, gold=None, append=False):
        client = FakeClient(responses)
        ctrl = day7.ClaudeController(client, max_searches=max_searches, book_titles={3: "Azkaban"}, retry_waits=(),
                                     append_missing_citations=append)
        retr = day7.AgentRetriever(day7.BM25(CHUNKS), 3, False, BY_ID, day7.neighbour_index(CHUNKS))
        q = {"id": "x", "split": "dev", "question": question, "answerable": answerable,
             "expected_answer": "ZZ-SENTINEL-ANSWER",
             "gold": gold or [{"source_id": SRC, "page": 12, "evidence": "made drills"}], **({"filter": flt} if flt else {})}
        cfg = {"max_searches": max_searches, "budget": 4000}
        return day7.run_case(q, ctrl, retr, cfg, {"use_filter": True}, BY_ID), client

    def test_search_then_cited_answer(self):
        r, client = self.run_q([response(tool("search_book", query="Grunnings firm")),
                                response(tool("submit_answer", status="answered", answer="Drills [C-b1-c01-0000].",
                                              citations=["C-b1-c01-0000"]))])
        self.assertEqual((r["status"], r["citations"], r["n_searches"]), ("answered", ["C-b1-c01-0000"], 1))
        self.assertTrue(r["context_has_evidence"])
        self.assertEqual(r["usage"]["input_tokens"], 200)
        self.assertIn("C-b1-c01-0000", client.prompts[1])          # 2nd call sees the cumulative context
        self.assertEqual(client.kwargs["output_config"], {"effort": "medium"})
        self.assertNotIn("ZZ-SENTINEL", " ".join(client.prompts) + client.kwargs["system"])  # benchmark answer never sent
        public = day7.public_trace(r)
        self.assertNotIn("context", public["events"][0])
        self.assertIn("snippet", public["events"][0]["results"][0])

    def test_unseen_citation_is_dropped_and_answer_abstains(self):
        r, _ = self.run_q([response(tool("search_book", query="Grunnings")),
                           response(tool("submit_answer", status="answered", answer="Drills [C-b3-c11-0037].",
                                         citations=["C-b3-c11-0037"]))])
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["llm_calls"][-1]["dropped_citations"], ["C-b3-c11-0037"])

    def test_listed_citation_without_inline_marker(self):
        steps = lambda: [response(tool("search_book", query="Grunnings")),  # noqa: E731
                         response(tool("submit_answer", status="answered", answer="Drills.", citations=["C-b1-c01-0000"]))]
        r, _ = self.run_q(steps())                      # A0-A2 behaviour: starter rule -> abstain
        self.assertEqual(r["status"], "abstain")
        r, _ = self.run_q(steps(), append=True)         # A3: listed in-context IDs rendered inline
        self.assertEqual((r["status"], r["answer"]), ("answered", "Drills. [C-b1-c01-0000]"))
        self.assertEqual(r["llm_calls"][-1]["citations_appended"], ["C-b1-c01-0000"])
        r, _ = self.run_q([response(tool("search_book", query="Grunnings")),  # IDs not in context stay rejected
                           response(tool("submit_answer", status="answered", answer="X.", citations=["C-b3-c11-0037"]))],
                          append=True)
        self.assertEqual(r["status"], "abstain")

    def test_insufficient_evidence_abstains(self):
        r, client = self.run_q([response(tool("search_book", query="Triwizard winner")),
                                response(tool("submit_answer", status="insufficient_evidence", answer="Not stated.",
                                              citations=[]))], answerable=False)
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["llm_calls"][-1]["model_status"], "insufficient_evidence")
        self.assertIsNone(r["context_has_evidence"])

    def test_search_limit_and_filter(self):
        r, client = self.run_q([response(tool("search_book", query="Firebolt")),
                                response(tool("search_book", query="Firebolt jinx"))], max_searches=1, flt={"book_no": 3})
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["events"][-1]["event"], "search_limit")
        self.assertEqual(r["events"][0]["filter"], {"book_no": 3})
        self.assertIn("reading Book 3", client.prompts[0])
        self.assertIn("Searches left: 0. Call submit_answer now.", client.prompts[1])

    def test_provider_error_recorded_not_hidden(self):
        import anthropic
        import httpx
        r, _ = self.run_q([anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))])
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["events"][-1]["event"], "provider_error")

    def test_refusal_is_an_error(self):
        r, _ = self.run_q([response(stop="refusal")])
        self.assertEqual((r["status"], r["events"][-1]["type"]), ("error", "ModelStop"))


if __name__ == "__main__":
    unittest.main()
