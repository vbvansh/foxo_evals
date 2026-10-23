import json

import pytest
from pydantic import BaseModel

from foxo_evals.llm.client import (
    InvalidOutputError,
    LLMClient,
    MissingRecordingError,
    cost_usd,
    extract_json,
    prune_recordings,
    recording_key,
)
from foxo_evals.tracing import Tracer


class Answer(BaseModel):
    value: int


class FakeAPI:
    """Stands in for the HTTP call: returns queued replies and remembers each request."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, payload: dict, session_id: str) -> dict:
        self.requests.append(payload)
        return {
            "choices": [{"message": {"content": self.replies.pop(0)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }


def no_network(payload: dict, session_id: str) -> dict:
    raise AssertionError("replay mode must not call the API")


def make_client(tmp_path, mode, post) -> LLMClient:
    return LLMClient("deepseek-v4.1-flash", Tracer(), mode=mode, recordings_dir=tmp_path, post=post)


def ask(client: LLMClient):
    return client.complete_json("Return the value.", "value is 7", Answer, agent="test", prompt_version="v1")


def test_replay_without_recording_names_the_case_and_agent(tmp_path):
    client = make_client(tmp_path, "replay", no_network)
    with client.tracer.trace("CASE-1"), pytest.raises(MissingRecordingError, match="CASE-1.*'test'"):
        ask(client)


def test_record_then_replay_gives_the_same_answer_offline(tmp_path):
    answer, usage = ask(make_client(tmp_path, "record", FakeAPI('{"value": 7}')))
    assert len(list(tmp_path.rglob("*.json"))) == 1

    replayed, replay_usage = ask(make_client(tmp_path, "replay", no_network))
    assert replayed == answer == Answer(value=7)
    assert replay_usage == usage


def test_record_mode_reuses_existing_recordings(tmp_path):
    ask(make_client(tmp_path, "record", FakeAPI('{"value": 7}')))
    api = FakeAPI('{"value": 8}')
    assert ask(make_client(tmp_path, "record", api))[0].value == 7
    assert api.requests == []


def test_live_mode_does_not_write_recordings(tmp_path):
    ask(make_client(tmp_path, "live", FakeAPI('{"value": 7}')))
    assert list(tmp_path.rglob("*.json")) == []


def test_invalid_output_is_retried_once_with_the_error(tmp_path):
    api = FakeAPI('{"value": "seven"}', '{"value": 7}')
    answer, usage = ask(make_client(tmp_path, "live", api))
    assert answer.value == 7
    assert len(api.requests) == 2
    retry_message = api.requests[1]["messages"][-1]["content"]
    assert "did not match" in retry_message and "value" in retry_message
    assert usage.tokens_in == 200 and usage.tokens_out == 40


def test_invalid_output_twice_raises_typed_error(tmp_path):
    client = make_client(tmp_path, "live", FakeAPI("not json", '{"wrong": 1}'))
    with pytest.raises(InvalidOutputError):
        ask(client)
    [span] = client.tracer.spans
    assert span.status == "error" and span.tokens_in == 200


def test_request_uses_temperature_zero_and_json_mode(tmp_path):
    api = FakeAPI('{"value": 7}')
    ask(make_client(tmp_path, "live", api))
    assert api.requests[0]["temperature"] == 0
    assert api.requests[0]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in api.requests[0]["messages"][0]["content"]


def test_llm_call_span_records_usage(tmp_path):
    client = make_client(tmp_path, "live", FakeAPI('{"value": 7}'))
    ask(client)
    [span] = client.tracer.spans
    assert (span.step, span.model, span.prompt_version) == ("llm_call", "deepseek-v4.1-flash", "v1")
    assert span.cost_usd == pytest.approx(cost_usd("deepseek-v4.1-flash", 100, 20))


def test_cost_uses_price_table():
    assert cost_usd("deepseek-v4.1-flash", 1_000_000, 1_000_000) == pytest.approx(1.50)
    assert cost_usd("unknown-model", 1000, 1000) == 0


def test_recording_key_changes_with_prompt_version_and_model():
    messages = [{"role": "user", "content": "hi"}]
    base = recording_key("doctor", "v1", "m1", messages)
    assert recording_key("doctor", "v2", "m1", messages) != base
    assert recording_key("doctor", "v1", "m2", messages) != base
    assert recording_key("doctor", "v1", "m1", messages) == base


def test_extract_json_strips_fences_and_chatter():
    assert json.loads(extract_json('Sure!\n```json\n{"value": 7}\n```')) == {"value": 7}


def test_prune_keeps_only_recordings_a_run_used(tmp_path):
    recordings = tmp_path / "recordings"
    old = LLMClient("deepseek-v4.1-flash", Tracer(), mode="record", recordings_dir=recordings, post=FakeAPI('{"value": 1}'))
    old.complete_json("Old prompt.", "value is 1", Answer, agent="test", prompt_version="v1")
    tracer = Tracer(tmp_path / "traces.jsonl")
    new = LLMClient("deepseek-v4.1-flash", tracer, mode="record", recordings_dir=recordings, post=FakeAPI('{"value": 2}'))
    with tracer.trace("CASE-1"):
        new.complete_json("New prompt.", "value is 2", Answer, agent="test", prompt_version="v1")

    removed = prune_recordings([tmp_path / "traces.jsonl"], recordings)
    assert len(removed) == 1
    assert [p.stem for p in recordings.rglob("*.json")] == tracer.spans[0].recording_keys
