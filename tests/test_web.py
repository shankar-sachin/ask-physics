import json

from askphysics.web import ask, info


def test_answered_question_is_display_ready() -> None:
    payload = json.loads(
        ask("How fast does a falling object hit the ground if it is dropped from 20 m?")
    )
    answer, display = payload["answer"], payload["display"]
    assert answer["status"] == "answered"
    assert display["result"] == {"value": "19.8057", "unit": "m/s"}
    assert display["equations"][0]["id"] == "kin_v_squared"
    assert display["equations"][0]["math"] == "v² = v₀² + 2·a·d"
    assert display["equations"][0]["source"]
    assert {"symbol": "v₀", "value": "0", "unit": "m/s", "origin": "assumption"} in display[
        "inputs"
    ]


def test_refusal_splits_reason_from_redirect() -> None:
    payload = json.loads(ask("How much does the color blue weigh?"))
    answer, display = payload["answer"], payload["display"]
    assert answer["status"] == "refused"
    assert answer["redirect"]
    assert display["why"] and answer["redirect"] not in display["why"]
    assert display["result"] is None


def test_partial_answer_reports_the_failed_stage() -> None:
    payload = json.loads(ask("How many rubber ducks would it take to stop a freight train?"))
    assert payload["answer"]["status"] == "degraded"
    assert any("plan stage failed" in c for c in payload["answer"]["caveats"])


def test_info_reports_version_and_data() -> None:
    data = json.loads(info())
    assert data["version"]
    assert data["data"]["equations"] > 0
