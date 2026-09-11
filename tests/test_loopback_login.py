import threading
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from indexrag.providers.orcarouter import CallbackReceiver, begin_login


def test_loopback_authorization_url():
    with CallbackReceiver() as receiver:
        verifier, url = begin_login(receiver.callback_url, receiver.state)
        params = parse_qs(urlsplit(url).query)
        assert params["callback_url"] == [receiver.callback_url]
        assert urlsplit(params["callback_url"][0]).hostname == "127.0.0.1"
        assert params["state"] == [receiver.state]
        assert params["code_challenge_method"] == ["S256"]
        assert verifier not in url


def test_callback_rejects_wrong_state_then_accepts_code():
    with CallbackReceiver() as receiver:
        result = []
        thread = threading.Thread(target=lambda: result.append(receiver.wait(5)))
        thread.start()
        bad = httpx.get(receiver.callback_url, params={"state": "wrong", "code": "bad"}, trust_env=False)
        assert bad.status_code == 400
        good = httpx.get(receiver.callback_url, params={"state": receiver.state, "code": "one-time"}, trust_env=False)
        assert good.status_code == 200
        assert "one-time" not in good.text
        thread.join(6)
        assert result == ["one-time"]
    # The listening socket has been released.
    with pytest.raises(httpx.ConnectError):
        httpx.get(receiver.callback_url, trust_env=False)


def test_denial_and_timeout():
    with CallbackReceiver() as receiver:
        result = []

        def wait():
            try:
                receiver.wait(5)
            except ValueError as exc:
                result.append(str(exc))

        thread = threading.Thread(target=wait)
        thread.start()
        httpx.get(
            receiver.callback_url + "?" + urlencode({"state": receiver.state, "error": "access_denied"}),
            trust_env=False,
        )
        thread.join(6)
        assert result == ["OrcaRouter authorization was declined."]
    with CallbackReceiver() as receiver, pytest.raises(ValueError, match="expired"):
        receiver.wait(0)


@pytest.mark.parametrize("success", [True, False])
def test_auto_close_only_after_successful_exchange(success):
    with CallbackReceiver() as receiver:
        completed = []
        errors = []

        def exchange(code):
            assert code == "valid-code"
            if not success:
                raise ValueError("Exchange failed")
            completed.append("saved")

        def wait():
            try:
                receiver.wait(5, on_code=exchange)
            except ValueError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=wait)
        thread.start()
        response = httpx.get(
            receiver.callback_url, params={"state": receiver.state, "code": "valid-code"}, trust_env=False
        )
        thread.join(6)
        assert not thread.is_alive()
        if success:
            assert completed == ["saved"]
            assert response.status_code == 200
            assert "window.close()" in response.text
            assert "history.replaceState" in response.text
            assert "credentials have been saved" in response.text
        else:
            assert errors == ["Exchange failed"]
            assert response.status_code == 400
            assert "window.close()" not in response.text
        assert "valid-code" not in response.text
