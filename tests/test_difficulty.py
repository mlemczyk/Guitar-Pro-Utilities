import json

from gp_fixture import build_gp, four_quarters
from gp_utils.difficulty import MEASURE_VERSION, measure, note_features, refinger
from gp_utils.difficulty import metrics


def test_weights_file_is_what_the_model_uses():
    w = json.loads(metrics.WEIGHTS_PATH.read_text(encoding="utf-8"))
    assert MEASURE_VERSION == w["measure_version"]
    assert metrics.V11_INTERCEPT == w["v11"]["intercept"]
    assert metrics.V11_COEF == w["v11"]["coef"]
    assert set(metrics.V11_COEF) <= set(note_features([{"start": 0, "midi": 33, "string": 1, "fret": 0}])[0])


def test_measure_reports_speed_and_prediction(tmp_path):
    bars = [four_quarters(28), four_quarters(33)]
    row = measure(build_gp(tmp_path / "t.gp", bars, tempo=120))
    assert row["error"] == ""
    assert row["note_count"] == 8 and row["playing_bars"] == 2
    assert abs(row["notes_per_minute"] - 120.0) < 1e-9           # quarters at 120 BPM
    assert 0 < row["predicted_accuracy_pct"] < 100


def test_faster_part_predicts_lower(tmp_path):
    slow = [[("Quarter", 28 + (i % 5))] * 4 for i in range(8)]
    fast = [[("16th", 28 + (j % 5)) for j in range(16)] for _ in range(8)]
    p_slow = measure(build_gp(tmp_path / "s.gp", slow, tempo=100))["predicted_accuracy_pct"]
    p_fast = measure(build_gp(tmp_path / "f.gp", fast, tempo=180))["predicted_accuracy_pct"]
    assert p_fast < p_slow


def test_rests_only_is_an_error(tmp_path):
    row = measure(build_gp(tmp_path / "r.gp", [[("Whole", None)]]))
    assert row["error"] == "no bars with notes"


def test_refinger_keeps_pitches_playable():
    notes = [{"start": i * 0.25, "end": i * 0.25 + 0.2, "midi": m, "string": None, "fret": None}
             for i, m in enumerate([23, 28, 33, 38, 43, 47])]   # B0 up to B2
    out = refinger(notes)
    assert [n["midi"] for n in out["notes"]] == [23, 28, 33, 38, 43, 47]
    assert out["octave_shifted"] == 0
    assert all(n["fret"] >= 0 for n in out["notes"])
