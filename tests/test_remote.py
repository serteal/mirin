"""Unit tests for remote client helpers."""

from __future__ import annotations

import logging

from tinyinterp.server.remote import _RemoteModel


def test_remote_release_logs_os_errors(caplog) -> None:
    remote = object.__new__(_RemoteModel)
    remote._closed = False
    remote._request = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom"))

    with caplog.at_level(logging.DEBUG):
        remote._release_value("value-1")
        remote._release_grad("grad-1")

    messages = [record.getMessage() for record in caplog.records]
    assert any("value-1" in message for message in messages)
    assert any("grad-1" in message for message in messages)
