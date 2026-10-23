import pytest

from foxo_evals.tracing import Tracer, format_trace, load_spans, summarise


def test_nested_spans_record_their_parent():
    tracer = Tracer()
    with tracer.trace("CASE-1"):
        with tracer.span("doctor", "agent") as outer:
            with tracer.span("doctor", "llm_call") as inner:
                pass
    assert outer.parent_span_id is None
    assert inner.parent_span_id == outer.span_id
    assert {s.trace_id for s in tracer.spans} == {"CASE-1"}


def test_errors_are_recorded_and_re_raised():
    tracer = Tracer()
    with pytest.raises(ValueError), tracer.span("mood", "normalise_units"):
        raise ValueError("bad unit")
    [span] = tracer.spans
    assert span.status == "error"
    assert span.error == "ValueError: bad unit"


def test_span_ids_are_deterministic():
    def run() -> list[str]:
        tracer = Tracer()
        with tracer.trace("CASE-1"):
            for step in ("a", "b"):
                with tracer.span("ome", step):
                    pass
        return [s.span_id for s in tracer.spans]

    assert run() == run()
    assert len(set(run())) == 2


def test_pii_fields_and_phone_numbers_are_redacted():
    text = summarise({"full_name": "Asha Rao", "phone": "9876543210", "age": 41})
    assert "Asha" not in text and "9876543210" not in text and "41" in text
    assert summarise("call me on +91 98765 43210 please") == "call me on [phone] please"


def test_long_summaries_are_truncated():
    assert len(summarise("x" * 5000)) < 500


def test_trace_is_written_in_start_order_and_rendered_as_tree(tmp_path):
    path = tmp_path / "traces.jsonl"
    tracer = Tracer(path)
    with tracer.trace("CASE-1"):
        with tracer.span("doctor", "agent"):
            with tracer.span("doctor", "llm_call", model="m1", prompt_version="v2"):
                pass

    spans = load_spans(path)
    assert [s.step for s in spans] == ["agent", "llm_call"]
    lines = format_trace(spans).splitlines()
    assert lines[0].startswith(f"[{spans[0].span_id}] doctor > agent")
    assert lines[1].startswith(f"  [{spans[1].span_id}] doctor > llm_call")
    assert "m1 prompt v2" in lines[1]
