import json
import unittest

from unittest.mock import patch

from business_eval_mindie import (
    MindIEClient,
    ensure_local_endpoint,
    first_token_logprobs,
    packed_batches,
    parse_mindie_body,
    score_completion_choice,
)


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class MindIEProtocolTests(unittest.TestCase):
    def test_local_endpoint_guard(self):
        ensure_local_endpoint("http://127.0.0.1:1025/v1/completions", False)
        ensure_local_endpoint("http://localhost:1025/v1/completions", False)
        with self.assertRaises(ValueError):
            ensure_local_endpoint("https://example.com/v1/completions", False)

    def test_completion_logprobs_score(self):
        choice = {
            "text": "yes",
            "logprobs": {
                "tokens": ["yes"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{"yes": -0.1, "no": -2.1}],
            },
        }
        logprobs, generated = first_token_logprobs(choice)
        score, missing, unexpected = score_completion_choice(choice, -20.0)
        self.assertEqual(generated, "yes")
        self.assertEqual(logprobs["no"], -2.1)
        self.assertGreater(score, 0.8)
        self.assertEqual(missing, 0)
        self.assertFalse(unexpected)

    def test_json_and_sse_response_parsing(self):
        response = {"choices": [{"index": 0, "text": "yes"}]}
        encoded = json.dumps(response).encode()
        self.assertEqual(parse_mindie_body(encoded), response)
        sse = b"data: {\"choices\":[{\"index\":0,\"text\":\"yes\"}]}\n\ndata: [DONE]\n"
        self.assertEqual(parse_mindie_body(sse)["choices"][0]["text"], "yes")

    def test_batches_respect_count_and_character_limits(self):
        indexed = [(0, "a" * 4), (1, "b" * 4), (2, "c" * 4)]
        batches = list(packed_batches(indexed, batch_size=2, max_request_chars=8))
        self.assertEqual([len(batch) for batch in batches], [2, 1])

        concurrent_batches = list(
            packed_batches(
                indexed,
                batch_size=3,
                max_request_chars=8,
                enforce_total_chars=False,
            )
        )
        self.assertEqual([len(batch) for batch in concurrent_batches], [3])

    def test_concurrent_mode_sends_one_prompt_per_request(self):
        client = MindIEClient(
            endpoint="http://127.0.0.1:1025/v1/completions",
            model_name="qwen3-reranker-4b",
            api_key="",
            timeout=1,
            retries=0,
            top_logprobs=5,
            missing_logprob_floor=-20.0,
            extra_request={},
            request_mode="concurrent",
        )

        def fake_urlopen(req, timeout):
            payload = json.loads(req.data.decode())
            self.assertIsInstance(payload["prompt"], str)
            token = "yes" if "relevant" in payload["prompt"] else "no"
            top_logprobs = (
                {"yes": -0.1, "no": -2.1}
                if token == "yes"
                else {"yes": -2.1, "no": -0.1}
            )
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "text": token,
                            "logprobs": {
                                "tokens": [token],
                                "token_logprobs": [-0.1],
                                "top_logprobs": [top_logprobs],
                            },
                        }
                    ]
                }
            )

        with patch("business_eval_mindie.request.urlopen", side_effect=fake_urlopen):
            scores = client.score_batch(["relevant document", "unrelated document"])
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])
        self.assertEqual(client.stats.request_count, 2)


if __name__ == "__main__":
    unittest.main()
