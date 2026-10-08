# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the multimodal SGLang generation client."""

import base64
import importlib.util
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

_SCRIPT_PATH = (
    Path(__file__).parents[3]
    / "examples/speculative_decoding/distributed_generate/server_generate_vlm_sglang.py"
)
_spec = importlib.util.spec_from_file_location("server_generate_vlm_sglang", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
server_generate_vlm_sglang = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server_generate_vlm_sglang)


def test_resolve_media_path_supports_local_and_remote_media(tmp_path):
    """Existing local media and supported remote media remain usable."""

    local_media = tmp_path / "image.jpg"
    local_media.touch()
    remote_media = "https://example.com/image.jpg"

    assert server_generate_vlm_sglang._resolve_media_path("image.jpg", str(tmp_path), None) == str(
        local_media
    )
    assert (
        server_generate_vlm_sglang._resolve_media_path(remote_media, str(tmp_path), None)
        == remote_media
    )


def test_resolve_media_path_returns_none_and_warns_once_for_missing_media(monkeypatch, capsys):
    """Missing local media can fall back to video or be skipped by the caller."""

    monkeypatch.setattr(server_generate_vlm_sglang, "_UNRESOLVED_MEDIA_PATHS", set())

    assert server_generate_vlm_sglang._resolve_media_path("missing.mp4", None, None) is None
    assert server_generate_vlm_sglang._resolve_media_path("missing.mp4", None, None) is None

    assert capsys.readouterr().out == "WARNING: could not resolve media path: missing.mp4\n"


def test_openai_media_value_is_relative_to_the_media_root(tmp_path):
    """The local HTTP server exposes media files, not the container root."""

    media_root = tmp_path / "media"
    media_path = media_root / "videos" / "clip.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.touch()
    input_path = tmp_path / "input" / "private.json"
    input_path.parent.mkdir()
    input_path.touch()

    assert (
        server_generate_vlm_sglang._as_openai_media_value(
            str(media_path), "http://127.0.0.1:18080", str(media_root), None
        )
        == "http://127.0.0.1:18080/videos/clip.mp4"
    )
    assert server_generate_vlm_sglang._as_openai_media_value(
        str(input_path), "http://127.0.0.1:18080", str(media_root), None
    ) == str(input_path)


def test_openai_media_value_inlines_local_media_as_a_data_uri(tmp_path):
    """--media_inline makes a request self-contained, with no media server."""

    media = tmp_path / "frame.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n")

    value = server_generate_vlm_sglang._as_openai_media_value(
        str(media), None, None, None, media_inline=True
    )

    assert value.startswith("data:image/png;base64,")
    assert base64.b64decode(value.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\n"


def test_openai_media_value_inline_rejects_missing_media(tmp_path):
    """Inlining a file that is not there must fail loudly, not send an empty payload."""

    with pytest.raises(FileNotFoundError):
        server_generate_vlm_sglang._as_openai_media_value(
            str(tmp_path / "gone.png"), None, None, None, media_inline=True
        )


def test_openai_media_value_warns_once_when_passing_a_bare_local_path(monkeypatch, capsys):
    """Without a delivery option the path only resolves on servers that read local files.

    SGLang native does; vLLM does not unless started with --allowed-local-media-path. The
    behavior is kept for backward compatibility, so the warning is what makes an
    unfetchable request diagnosable instead of a confusing server-side error.
    """

    monkeypatch.setattr(server_generate_vlm_sglang, "_WARNED_LOCAL_MEDIA", False)

    assert server_generate_vlm_sglang._as_openai_media_value("/data/a.mp4", None, None, None) == (
        "/data/a.mp4"
    )
    assert server_generate_vlm_sglang._as_openai_media_value("/data/b.mp4", None, None, None) == (
        "/data/b.mp4"
    )

    assert capsys.readouterr().out.count("WARNING:") == 1


def test_openai_media_value_prefers_remote_urls_over_inlining(tmp_path):
    """An http(s)/data value is already fetchable and must pass through untouched."""

    remote = "https://example.com/clip.mp4"
    assert (
        server_generate_vlm_sglang._as_openai_media_value(remote, None, None, None, True) == remote
    )


@pytest.fixture
def openai_http_server(monkeypatch):
    """Receive actual CLI requests without running an inference engine."""
    received = []
    reply = {"status": 200, "body": {"choices": [{"message": {"content": "Fixture answer."}}]}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.connection.settimeout(5)
            received.append(
                (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            )
            body = json.dumps(reply["body"]).encode()
            self.send_response(reply["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", received, reply
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def _run_openai_cli(tmp_path, url, media_kind, *extra_args):
    media = tmp_path / ("frame.png" if media_kind == "image" else "clip.mp4")
    media.write_bytes(b"fixture media bytes")
    sample = {
        "id": "fixture",
        "messages": [{"role": "user", "content": "Describe this media."}],
        f"{media_kind}_path": media.name,
    }
    data = tmp_path / "prompts.jsonl"
    data.write_text(json.dumps(sample) + "\n")
    output = tmp_path / "output.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--api_mode",
            "openai",
            "--url",
            url,
            "--model_name",
            "fixture-model",
            "--data_path",
            str(data),
            "--output_path",
            str(output),
            "--media_root",
            str(tmp_path),
            "--media_inline",
            "--num_threads",
            "1",
            "--temperature",
            "0.25",
            "--max_tokens",
            "8",
            "--request_timeout",
            "5",
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (
        sample,
        media.read_bytes(),
        [json.loads(line) for line in output.read_text().splitlines()],
    )


@pytest.mark.parametrize("media_kind", ["image", "video"])
def test_openai_cli_sends_inline_media_over_http(tmp_path, openai_http_server, media_kind):
    """The real CLI sends inline media and writes the HTTP server's assistant response."""
    url, received, _ = openai_http_server
    sample, media, output = _run_openai_cli(tmp_path, url, media_kind)
    assert len(received) == 1
    path, payload = received[0]
    assert path == "/v1/chat/completions"
    assert payload["model"] == "fixture-model"
    assert payload["temperature"] == 0.25 and payload["max_tokens"] == 8
    (message,) = payload["messages"]
    assert message["role"] == "user"
    text, content = message["content"]
    assert text == {"type": "text", "text": sample["messages"][0]["content"]}
    kind = f"{media_kind}_url"
    assert content["type"] == kind
    prefix, encoded = content[kind]["url"].split(",", 1)
    assert prefix == f"data:{'image/png' if media_kind == 'image' else 'video/mp4'};base64"
    assert base64.b64decode(encoded, validate=True) == media
    assert output == [
        {
            "conversation_id": 0,
            "id": sample["id"],
            "messages": sample["messages"] + [{"role": "assistant", "content": "Fixture answer."}],
            f"{media_kind}_path": sample[f"{media_kind}_path"],
        }
    ]


def test_openai_cli_records_http_failure(tmp_path, openai_http_server):
    """An HTTP rejection is retained in output when empty conversations are requested."""
    url, received, reply = openai_http_server
    reply.update(status=400, body={"error": {"message": "media rejected"}})
    _, _, output = _run_openai_cli(tmp_path, url, "image", "--log_empty_conversations")
    assert len(received) == 1
    assert output[0]["messages"][-1] == {"role": "assistant", "content": ""}
    assert output[0]["generation_error"].startswith("HTTP 400:")
    assert "media rejected" in output[0]["generation_error"]
    assert output[1] == {"finished": True}
