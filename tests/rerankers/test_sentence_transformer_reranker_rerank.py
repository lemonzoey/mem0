"""Behavior tests for ``SentenceTransformerReranker.rerank`` on the normal (non-fallback) path.

The fallback path is covered by ``test_reranker_fallback_no_mutation.py`` and
``test_reranker_fallback_topk.py``. These tests assert the rerank *contract*
shared by every reranker — sorted by score descending, at most ``top_k``
results, each carrying ``rerank_score`` as a Python ``float`` with original
fields preserved, empty input short-circuits, and text extraction follows
``memory`` → ``text`` → ``content`` → ``str(doc)`` — rather than the
``CrossEncoder.predict`` call shape. ``CrossEncoder`` is mocked so no model is
loaded; numpy and plain-list score returns are both exercised since either is
a valid provider output.
"""

from unittest.mock import patch

import numpy as np

from mem0.configs.rerankers.sentence_transformer import (
    SentenceTransformerRerankerConfig,
)


class TestSentenceTransformerRerankNormalPath:
    def _build(self, mock_model, **overrides):
        kwargs = dict(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            api_key=None,
            top_k=None,
            device="cpu",
            batch_size=32,
            show_progress_bar=False,
        )
        kwargs.update(overrides)
        config = SentenceTransformerRerankerConfig(**kwargs)
        from mem0.reranker.sentence_transformer_reranker import (
            SentenceTransformerReranker,
        )

        return SentenceTransformerReranker(config), mock_model

    def test_empty_documents_short_circuits(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            reranker, _ = self._build(mock_model)

            assert reranker.rerank("query", []) == []
            mock_model.predict.assert_not_called()

    def test_configured_model_reaches_the_cross_encoder(self):
        # A user-visible behaviour: the model they configured is the one loaded.
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            self._build(mock_ce_cls.return_value)

            mock_ce_cls.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    def test_numpy_scores_sorted_descending(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
            reranker, _ = self._build(mock_model)

            docs = [
                {"memory": "a", "id": "0"},
                {"memory": "b", "id": "1"},
                {"memory": "c", "id": "2"},
            ]
            result = reranker.rerank("query", docs)

            assert [r["memory"] for r in result] == ["b", "c", "a"]
            assert [r["rerank_score"] for r in result] == [0.9, 0.5, 0.1]

    def test_scores_coerced_to_python_float(self):
        # numpy scalars must become plain floats so downstream JSON/serialization
        # does not silently break.
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.9, 0.5])
            reranker, _ = self._build(mock_model)

            result = reranker.rerank("query", [{"memory": "a"}, {"memory": "b"}])

            assert isinstance(result[0]["rerank_score"], float)
            assert type(result[0]["rerank_score"]) is float

    def test_plain_list_scores_handled_without_numpy(self):
        # CrossEncoder.predict may return a plain list; ranking and float
        # coercion must still hold.
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = [0.1, 0.9, 0.5]
            reranker, _ = self._build(mock_model)

            result = reranker.rerank("query", [{"memory": "a"}, {"memory": "b"}, {"memory": "c"}])

            assert [r["memory"] for r in result] == ["b", "c", "a"]
            assert isinstance(result[0]["rerank_score"], float)

    def test_results_carry_rerank_score_and_original_fields(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.1, 0.9])
            reranker, _ = self._build(mock_model)

            docs = [{"memory": "a", "id": "0"}, {"memory": "b", "id": "1"}]
            result = reranker.rerank("query", docs)

            assert result[0]["id"] == "1"
            assert result[0]["memory"] == "b"
            assert result[0]["rerank_score"] == 0.9

    def test_top_k_arg_truncates_results(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.9, 0.5, 0.1])
            reranker, _ = self._build(mock_model)

            result = reranker.rerank("query", [{"memory": f"doc{i}"} for i in range(3)], top_k=2)

            assert len(result) == 2
            assert [r["rerank_score"] for r in result] == [0.9, 0.5]

    def test_config_top_k_truncates_when_arg_not_provided(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.9, 0.5, 0.1])
            reranker, _ = self._build(mock_model, top_k=1)

            result = reranker.rerank("query", [{"memory": f"doc{i}"} for i in range(3)])

            assert len(result) == 1
            assert result[0]["rerank_score"] == 0.9

    def test_all_documents_returned_when_no_top_k(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.9, 0.5, 0.1])
            reranker, _ = self._build(mock_model)

            result = reranker.rerank("query", [{"memory": f"doc{i}"} for i in range(3)])

            assert len(result) == 3

    def test_text_extraction_prefers_memory_then_text_then_content(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.5])
            reranker, _ = self._build(mock_model)

            reranker.rerank("query", [{"memory": "from-memory", "text": "from-text", "content": "from-content"}])

            # The extracted text is what gets paired with the query for scoring.
            assert mock_model.predict.call_args[0][0] == [["query", "from-memory"]]

    def test_text_extraction_falls_back_to_text(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.5])
            reranker, _ = self._build(mock_model)

            reranker.rerank("query", [{"text": "from-text", "content": "from-content"}])

            assert mock_model.predict.call_args[0][0] == [["query", "from-text"]]

    def test_text_extraction_falls_back_to_content(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.5])
            reranker, _ = self._build(mock_model)

            reranker.rerank("query", [{"content": "from-content"}])

            assert mock_model.predict.call_args[0][0] == [["query", "from-content"]]

    def test_text_extraction_falls_back_to_str_for_unknown_shape(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.5])
            reranker, _ = self._build(mock_model)

            reranker.rerank("query", [{"custom": "payload"}])

            assert mock_model.predict.call_args[0][0] == [["query", "{'custom': 'payload'}"]]

    def test_original_documents_not_mutated(self):
        with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder") as mock_ce_cls:
            mock_model = mock_ce_cls.return_value
            mock_model.predict.return_value = np.array([0.8])
            reranker, _ = self._build(mock_model)

            original = {"memory": "doc", "id": "123"}
            result = reranker.rerank("query", [original])

            assert "rerank_score" not in original
            assert result[0]["rerank_score"] == 0.8
