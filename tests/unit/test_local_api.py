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

    monkeypatch.setattr("urllib.request.urlopen", _refuse)
    assert api_is_up("http://127.0.0.1:8000") is False


def test_api_is_up_honours_a_patched_urllib_urlopen(monkeypatch):
    """Regression: ``from urllib.request import urlopen`` froze the real
    function at import, so UI tests patching ``urllib.request.urlopen`` still
    hit the real port and launched uvicorn from inside the test run."""
    seen = []

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def _fake(req, timeout):
        seen.append((req.get_method(), req.full_url))
        return _Ok()

    monkeypatch.setattr("urllib.request.urlopen", _fake)
    assert api_is_up("http://127.0.0.1:8000") is True
    assert seen == [("GET", "http://127.0.0.1:8000/health")]
