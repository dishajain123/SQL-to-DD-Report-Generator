from app.review.local_api import api_is_up, ensure_local_api


def test_ensure_local_api_does_nothing_when_health_responds(monkeypatch):
    monkeypatch.setattr("app.review.local_api.api_is_up", lambda *_a, **_k: True)
    assert ensure_local_api("http://127.0.0.1:8000") == ""


def test_ensure_local_api_skips_remote_urls(monkeypatch):
    monkeypatch.setattr("app.review.local_api.api_is_up", lambda *_a, **_k: False)
    assert ensure_local_api("http://api.example.com:8000") == ""


def test_api_is_up_is_false_when_connection_is_refused(monkeypatch):
    def _refuse(*_args, **_kwargs):
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr("app.review.local_api.urlopen", _refuse)
    assert api_is_up("http://127.0.0.1:8000") is False
