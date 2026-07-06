# -*- coding: utf-8 -*-
"""model/inference/tei_client.py 单元测试：重试、空输入短路、响应字段校验。

回归测试审查报告 L19：`embed`/`embed_batch`/`rerank` 此前仅调用
`raise_for_status()`，未捕获网络超时/JSON 解析失败，也无重试；
`rerank([], [])` 会向 TEI 发空 texts；`rerank_scores` 直接 `item["index"]`
无校验。
"""
import json
import requests
import pytest

from model.inference.tei_client import TEIClient


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data) if json_data else "{}"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return self._json_data


class TestEmptyInputShortCircuit:
    def test_rerank_empty_texts_returns_empty_without_network_call(self, monkeypatch):
        client = TEIClient()
        calls = []
        monkeypatch.setattr(client.session, "post", lambda *a, **kw: calls.append(1))
        assert client.rerank("query", []) == []
        assert calls == []

    def test_rerank_scores_empty_texts_returns_empty(self, monkeypatch):
        client = TEIClient()
        calls = []
        monkeypatch.setattr(client.session, "post", lambda *a, **kw: calls.append(1))
        assert client.rerank_scores("query", []) == []
        assert calls == []

    def test_embed_batch_empty_texts_returns_empty_without_network_call(self, monkeypatch):
        client = TEIClient()
        calls = []
        monkeypatch.setattr(client.session, "post", lambda *a, **kw: calls.append(1))
        assert client.embed_batch([]) == []
        assert calls == []


class TestRetryOnTransientFailure:
    def test_embed_retries_on_timeout_then_succeeds(self, monkeypatch):
        client = TEIClient(max_retries=2, retry_backoff_seconds=0.01)
        calls = {"count": 0}

        def _fake_post(url, json=None, timeout=None):
            calls["count"] += 1
            if calls["count"] < 3:
                raise requests.exceptions.Timeout("模拟超时")
            return _FakeResponse([[0.1, 0.2, 0.3]])

        monkeypatch.setattr(client.session, "post", _fake_post)
        result = client.embed("你好")
        assert result == [0.1, 0.2, 0.3]
        assert calls["count"] == 3  # 前两次超时失败 + 第三次成功

    def test_embed_raises_after_exhausting_retries(self, monkeypatch):
        client = TEIClient(max_retries=1, retry_backoff_seconds=0.01)
        calls = {"count": 0}

        def _fake_post(url, json=None, timeout=None):
            calls["count"] += 1
            raise requests.exceptions.ConnectionError("模拟连接失败")

        monkeypatch.setattr(client.session, "post", _fake_post)
        with pytest.raises(requests.exceptions.ConnectionError):
            client.embed("你好")
        assert calls["count"] == 2  # 首次 + 1 次重试

    def test_http_4xx_not_retried(self, monkeypatch):
        """【修复 N19】4xx 状态码不应重试（请求本身有问题，重试无意义）。"""
        import json as _json
        client = TEIClient(max_retries=3, retry_backoff_seconds=0.01)
        calls = {"count": 0}

        def _fake_post(url, json=None, timeout=None):
            calls["count"] += 1
            return _FakeResponse({"error": "bad request"}, status_code=400)

        monkeypatch.setattr(client.session, "post", _fake_post)
        with pytest.raises(requests.exceptions.HTTPError):
            client.embed("你好")
        assert calls["count"] == 1  # 4xx 不重试

    def test_http_5xx_retried(self, monkeypatch):
        """【修复 N19】5xx 状态码应重试（服务端瞬时错误）。"""
        client = TEIClient(max_retries=2, retry_backoff_seconds=0.01)
        calls = {"count": 0}

        def _fake_post(url, json=None, timeout=None):
            calls["count"] += 1
            if calls["count"] < 3:
                return _FakeResponse({"error": "server error"}, status_code=503)
            return _FakeResponse([[0.1, 0.2]])

        monkeypatch.setattr(client.session, "post", _fake_post)
        result = client.embed("你好")
        assert result == [0.1, 0.2]
        assert calls["count"] == 3  # 两次 503 重试 + 第三次成功


class TestRerankScoresValidation:
    def test_missing_index_field_raises_clear_error(self, monkeypatch):
        client = TEIClient()
        monkeypatch.setattr(
            client, "rerank", lambda query, texts, return_text=False: [{"score": 0.9}]
        )
        with pytest.raises(ValueError, match="缺少 index/score"):
            client.rerank_scores("query", ["doc1"])

    def test_out_of_range_index_raises_clear_error(self, monkeypatch):
        client = TEIClient()
        monkeypatch.setattr(
            client, "rerank", lambda query, texts, return_text=False: [{"index": 5, "score": 0.9}]
        )
        with pytest.raises(ValueError, match="越界"):
            client.rerank_scores("query", ["doc1"])

    def test_valid_response_restores_original_order(self, monkeypatch):
        client = TEIClient()
        monkeypatch.setattr(
            client, "rerank",
            lambda query, texts, return_text=False: [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.5}],
        )
        scores = client.rerank_scores("query", ["doc0", "doc1"])
        assert scores == [0.5, 0.9]
