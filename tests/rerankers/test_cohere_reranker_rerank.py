"""Behavior tests for ``CohereReranker.rerank`` on the normal (non-fallback) path.

The fallback path is covered by ``test_reranker_fallback_no_mutation.py`` and
``test_reranker_fallback_topk.py``. These tests assert the rerank *contract*
shared by every reranker — sorted by score descending, at most ``top_k`` results,
each carrying ``rerank_score`` with original fields preserved, empty input
short-circuits, and text extraction follows ``memory`` → ``text`` → ``content``
→ ``str(doc)`` — rather than Cohere's internal call shape. The fake client
mimics the real Cohere API: it honours ``top_n`` by returning only that many
results, so truncation is verified through behaviour, not argument matching.
"""

from dataclasses import dataclass
from typing import List

from mem0.configs.rerankers.cohere import CohereRerankerConfig


@dataclass
class _CohereResult:
    index: int
    relevance_score: float


@dataclass
class _CohereResponse:
    results: List[_CohereResult]


def _cohere_api(**kwargs):
    """Mimic Cohere's rerank API: truncate to ``top_n`` and return by index."""
    n = kwargs["top_n"]
    return _CohereResponse(results=[_CohereResult(index=i, relevance_score=1.0 - i * 0.1) for i in range(n)])


class TestCohereRerankNormalPath:
    def test_empty_documents_short_circuits(self, mock_cohere):
        module, fake_client = mock_cohere
        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))

        assert reranker.rerank("query", []) == []
        fake_client.rerank.assert_not_called()

    def test_configured_model_reaches_the_api(self, mock_cohere):
        # A user-visible behaviour: the model they configured is the one reranked
        # against. Configuring the wrong model silently changes results.
        module, fake_client = mock_cohere
        fake_client.rerank.return_value = _CohereResponse(results=[])

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key", model="rerank-v3.5"))
        reranker.rerank("query", [{"memory": "a"}])

        assert fake_client.rerank.call_args.kwargs["model"] == "rerank-v3.5"

    def test_results_sorted_by_score_descending(self, mock_cohere):
        module, fake_client = mock_cohere
        # API returns in arbitrary order; results must come out score-descending.
        fake_client.rerank.return_value = _CohereResponse(
            results=[
                _CohereResult(index=1, relevance_score=0.9),
                _CohereResult(index=0, relevance_score=0.5),
                _CohereResult(index=2, relevance_score=0.1),
            ]
        )

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        docs = [{"memory": "a", "id": "0"}, {"memory": "b", "id": "1"}, {"memory": "c", "id": "2"}]
        result = reranker.rerank("query", docs)

        assert [r["rerank_score"] for r in result] == [0.9, 0.5, 0.1]

    def test_results_carry_rerank_score_and_original_fields(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.return_value = _CohereResponse(results=[_CohereResult(index=2, relevance_score=0.9)])

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        docs = [{"memory": "a", "id": "0"}, {"memory": "b", "id": "1"}, {"memory": "c", "id": "2"}]
        result = reranker.rerank("query", docs)

        assert result[0]["id"] == "2"
        assert result[0]["memory"] == "c"
        assert result[0]["rerank_score"] == 0.9

    def test_top_k_arg_truncates_results(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.side_effect = _cohere_api

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        docs = [{"memory": f"doc{i}"} for i in range(3)]

        result = reranker.rerank("query", docs, top_k=2)

        assert len(result) == 2
        assert all("rerank_score" in r for r in result)

    def test_config_top_k_truncates_when_arg_not_provided(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.side_effect = _cohere_api

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key", top_k=1))
        docs = [{"memory": f"doc{i}"} for i in range(3)]

        assert len(reranker.rerank("query", docs)) == 1

    def test_all_documents_returned_when_no_top_k(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.side_effect = _cohere_api

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        docs = [{"memory": f"doc{i}"} for i in range(3)]

        assert len(reranker.rerank("query", docs)) == 3

    def test_text_extraction_prefers_memory_then_text_then_content(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.return_value = _CohereResponse(results=[_CohereResult(index=0, relevance_score=0.5)])

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"memory": "from-memory", "text": "from-text", "content": "from-content"}])

        assert fake_client.rerank.call_args.kwargs["documents"] == ["from-memory"]

    def test_text_extraction_falls_back_to_text(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.return_value = _CohereResponse(results=[_CohereResult(index=0, relevance_score=0.5)])

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"text": "from-text", "content": "from-content"}])

        assert fake_client.rerank.call_args.kwargs["documents"] == ["from-text"]

    def test_text_extraction_falls_back_to_content(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.return_value = _CohereResponse(results=[_CohereResult(index=0, relevance_score=0.5)])

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"content": "from-content"}])

        assert fake_client.rerank.call_args.kwargs["documents"] == ["from-content"]

    def test_text_extraction_falls_back_to_str_for_unknown_shape(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.return_value = _CohereResponse(results=[_CohereResult(index=0, relevance_score=0.5)])

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"custom": "payload"}])

        assert fake_client.rerank.call_args.kwargs["documents"] == ["{'custom': 'payload'}"]

    def test_original_documents_not_mutated(self, mock_cohere):
        module, fake_client = mock_cohere
        fake_client.rerank.return_value = _CohereResponse(results=[_CohereResult(index=0, relevance_score=0.8)])

        reranker = module.CohereReranker(CohereRerankerConfig(api_key="test-key"))
        original = {"memory": "doc", "id": "123"}
        result = reranker.rerank("query", [original])

        assert "rerank_score" not in original
        assert result[0]["rerank_score"] == 0.8
