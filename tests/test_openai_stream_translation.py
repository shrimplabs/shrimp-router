"""An OpenAI-format backend's stream must reach /v1/messages clients as
Anthropic SSE (it used to pass through untranslated: empty replies)."""
import asyncio
import json

from shrimp_router.format_normalizer import openai_stream_to_anthropic


class FakeStream:
    def __init__(self, pieces):
        self.pieces = pieces
        self.closed = False

    async def aiter_text(self):
        for piece in self.pieces:
            yield piece

    async def aclose(self):
        self.closed = True


def chunk(**delta):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}) + "\n\n"


def events(raw: bytes):
    out = []
    for block in raw.decode().strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def run(pieces):
    stream = FakeStream(pieces)

    async def collect():
        return b"".join([c async for c in openai_stream_to_anthropic(stream, "qwen-flash-next")])
    return events(asyncio.run(collect())), stream


def test_reasoning_then_answer_become_thinking_and_text_blocks():
    finish = "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
    usage = "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 58, "completion_tokens": 31}}) + "\n\n"
    whole = chunk(reasoning_content="We need ") + chunk(reasoning_content="PONG.") + chunk(content="PO") + chunk(content="NG") + finish + usage + "data: [DONE]\n\n"
    # Split mid-line, as network reads do.
    evs, stream = run([whole[:37], whole[37:150], whole[150:]])
    kinds = [e for e, _ in evs]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    thinking = "".join(d["delta"]["thinking"] for e, d in evs if e == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
    text = "".join(d["delta"]["text"] for e, d in evs if e == "content_block_delta" and d["delta"]["type"] == "text_delta")
    assert thinking == "We need PONG." and text == "PONG"
    starts = [d["content_block"]["type"] for e, d in evs if e == "content_block_start"]
    assert starts == ["thinking", "text"]
    assert kinds.count("content_block_stop") == 2
    delta = next(d for e, d in evs if e == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn" and delta["usage"]["output_tokens"] == 31
    assert stream.closed


def test_answer_without_reasoning_and_length_stop():
    finish = "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": "length"}]}) + "\n\n"
    evs, _ = run([chunk(content="hi") + finish + "data: [DONE]\n\n"])
    text = "".join(d["delta"]["text"] for e, d in evs if e == "content_block_delta")
    assert text == "hi!"
    assert next(d for e, d in evs if e == "message_delta")["delta"]["stop_reason"] == "max_tokens"


def test_streamed_openai_request_asks_for_usage():
    from shrimp_router.app import _anthropic_to_openai
    out = _anthropic_to_openai({"stream": True, "max_tokens": 10, "messages": [{"role": "user", "content": "x"}]}, "m")
    assert out["stream"] is True and out["stream_options"] == {"include_usage": True}
    assert "stream_options" not in _anthropic_to_openai({"messages": []}, "m")


def test_mistral_typed_content_parts_become_thinking_and_text():
    # Mistral (GLM 5.3) streams reasoning as typed part lists, not strings.
    import asyncio
    import json as _json
    from shrimp_router.format_normalizer import _split_content, openai_stream_to_anthropic

    assert _split_content([{"type": "thinking", "thinking": [{"type": "text", "text": "hm"}]},
                           {"type": "text", "text": "hi"}]) == ("hm", "hi")
    chunks = [
        {"choices": [{"delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]}]}}]},
        {"choices": [{"delta": {"content": [{"type": "text", "text": "[TOOL_CALL]"}]}}]},
        {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]},
    ]

    class Resp:
        async def aiter_text(self):
            for c in chunks:
                yield "data: " + _json.dumps(c) + "\n\n"
            yield "data: [DONE]\n\n"

        async def aclose(self):
            pass

    async def collect():
        return [e async for e in openai_stream_to_anthropic(Resp(), "m")]

    out = "".join(e.decode() if isinstance(e, bytes) else e for e in asyncio.run(collect()))
    assert '"thinking": "plan"' in out and '"text": "[TOOL_CALL]"' in out and '"text": "done"' in out
