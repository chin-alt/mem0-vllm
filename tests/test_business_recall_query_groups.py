import json
import sys
import tempfile
import unittest

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_business import (  # noqa: E402
    GroundTruthItem,
    build_scoring_inputs,
    load_recall_results,
    select_recall_top_k,
)


class BusinessRecallQueryGroupTests(unittest.TestCase):
    def test_query_keyed_json_is_loaded_completely_and_kept_grouped(self):
        payload = {
            "query-a": [
                {"index": 1, "id": "a-1", "text": "document a1"},
                {"index": 2, "id": "a-2", "text": "document a2"},
            ],
            "query-b": [
                {"index": 1, "id": "b-1", "text": "document b1"},
                {"index": 2, "id": "b-2", "text": "document b2"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            recall_path = Path(temp_dir) / "recall.json"
            recall_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            recall = load_recall_results(
                recall_path,
                id_key="id",
                text_key="text",
            )

        ground_truth = {
            "query-a": GroundTruthItem(query="query-a", doc_ids=["a-1"]),
            "query-b": GroundTruthItem(query="query-b", doc_ids=["b-1"]),
        }
        _inputs, mapping, skipped = build_scoring_inputs(
            recall,
            ground_truth,
            instruction="instruction",
        )

        self.assertEqual(skipped, 0)
        self.assertEqual([row["query"] for row in mapping], [
            "query-a",
            "query-a",
            "query-b",
            "query-b",
        ])
        self.assertEqual([row["doc_id"] for row in mapping], [
            "a-1",
            "a-2",
            "b-1",
            "b-2",
        ])

    def test_recall_top_k_uses_json_index_not_array_order(self):
        payload = {
            "query-a": [
                {"index": 21, "id": "a-21", "text": "document 21"},
                {"index": 2, "id": "a-2", "text": "document 2"},
                {"index": 1, "id": "a-1", "text": "document 1"},
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            recall_path = Path(temp_dir) / "recall.json"
            recall_path.write_text(json.dumps(payload), encoding="utf-8")
            recall = load_recall_results(recall_path, id_key="id", text_key="text")

        selected = select_recall_top_k(recall, 2)
        self.assertEqual(
            [doc["doc_id"] for doc in selected["query-a"]],
            ["a-1", "a-2"],
        )
        self.assertEqual(
            [doc["source_rank"] for doc in selected["query-a"]],
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
