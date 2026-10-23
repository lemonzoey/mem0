"""Behavior tests for ``ZeroEntropyReranker.rerank`` on the normal (non-fallback) path.

The fallback path is covered by ``test_reranker_fallback_no_mutation.py`` and
``test_reranker_fallback_topk.py``. These tests assert the rerank *contract*
shared by every reranker — sorted by score descending, at most ``top_k``
results, each carrying ``rerank_score`` with original fields preserved, empty
input short-circuits, and text extraction follows ``memory`` → ``text`` →
``content`` → ``str(doc)`` — rather than the ``client.models.rerank`` call
shape. ZeroEntropy's API does not take a ``top_k``, so truncation is applied
locally after sorting; the tests verify that behaviour, not the call args.
"""

from dataclasses import dataclass
from typing import List

from mem0.configs.rerankers.zero_entropy import ZeroEntropyRerankerConfig


@dataclass
class _ZEResult:
    index: int
    relevance_score: float


@dataclass
class _ZEResponse:
    results: List[_ZEResult]


class TestZeroEntropyRerankNormalPath:
    def test_empty_documents_short_circuits(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))

        assert reranker.rerank("query", []) == []
        fake_client.models.rerank.assert_not_called()

    def test_configured_model_reaches_the_api(self, mock_zero_entropy):
        # A user-visible behaviour: the model they configured is the one reranked
        # against. Configuring the wrong model silently changes results.
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(results=[])

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key", model="zerank-1"))
        reranker.rerank("query", [{"memory": "a"}])

        assert fake_client.models.rerank.call_args.kwargs["model"] == "zerank-1"

    def test_results_sorted_by_score_descending(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        # API returns results in arbitrary index/score order; the provider
        # re-sorts by rerank_score descending.
        fake_client.models.rerank.return_value = _ZEResponse(
            results=[
                _ZEResult(index=0, relevance_score=0.1),
                _ZEResult(index=1, relevance_score=0.9),
                _ZEResult(index=2, relevance_score=0.5),
            ]
        )

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        docs = [{"memory": "a", "id": "0"}, {"memory": "b", "id": "1"}, {"memory": "c", "id": "2"}]
        result = reranker.rerank("query", docs)

        assert [r["memory"] for r in result] == ["b", "c", "a"]
        assert [r["rerank_score"] for r in result] == [0.9, 0.5, 0.1]

    def test_results_carry_rerank_score_and_original_fields(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(results=[_ZEResult(index=1, relevance_score=0.9)])

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        docs = [{"memory": "a", "id": "0"}, {"memory": "b", "id": "1"}]
        result = reranker.rerank("query", docs)

        assert result[0]["id"] == "1"
        assert result[0]["memory"] == "b"
        assert result[0]["rerank_score"] == 0.9

    def test_top_k_arg_truncates_results(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(
            results=[_ZEResult(index=i, relevance_score=1.0 - i * 0.1) for i in range(3)]
        )

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        result = reranker.rerank("query", [{"memory": f"doc{i}"} for i in range(3)], top_k=2)

        assert len(result) == 2
        # Sorted descending, so the two highest scores survive.
        assert [r["rerank_score"] for r in result] == [1.0, 0.9]

    def test_config_top_k_truncates_when_arg_not_provided(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(
            results=[_ZEResult(index=i, relevance_score=1.0 - i * 0.1) for i in range(3)]
        )

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key", top_k=1))
        result = reranker.rerank("query", [{"memory": f"doc{i}"} for i in range(3)])

        assert len(result) == 1
        assert result[0]["rerank_score"] == 1.0

    def test_all_documents_returned_when_no_top_k(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(
            results=[_ZEResult(index=i, relevance_score=1.0 - i * 0.1) for i in range(3)]
        )

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        result = reranker.rerank("query", [{"memory": f"doc{i}"} for i in range(3)])

        assert len(result) == 3

    def test_text_extraction_prefers_memory_then_text_then_content(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(results=[_ZEResult(index=0, relevance_score=0.5)])

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"memory": "from-memory", "text": "from-text", "content": "from-content"}])

        assert fake_client.models.rerank.call_args.kwargs["documents"] == ["from-memory"]

    def test_text_extraction_falls_back_to_text(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(results=[_ZEResult(index=0, relevance_score=0.5)])

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"text": "from-text", "content": "from-content"}])

        assert fake_client.models.rerank.call_args.kwargs["documents"] == ["from-text"]

    def test_text_extraction_falls_back_to_content(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(results=[_ZEResult(index=0, relevance_score=0.5)])

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"content": "from-content"}])

        assert fake_client.models.rerank.call_args.kwargs["documents"] == ["from-content"]

    def test_text_extraction_falls_back_to_str_for_unknown_shape(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(results=[_ZEResult(index=0, relevance_score=0.5)])

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        reranker.rerank("query", [{"custom": "payload"}])

        assert fake_client.models.rerank.call_args.kwargs["documents"] == ["{'custom': 'payload'}"]

    def test_original_documents_not_mutated(self, mock_zero_entropy):
        module, fake_client = mock_zero_entropy
        fake_client.models.rerank.return_value = _ZEResponse(results=[_ZEResult(index=0, relevance_score=0.8)])

        reranker = module.ZeroEntropyReranker(ZeroEntropyRerankerConfig(api_key="test-key"))
        original = {"memory": "doc", "id": "123"}
        result = reranker.rerank("query", [original])

        assert "rerank_score" not in original
        assert result[0]["rerank_score"] == 0.8
