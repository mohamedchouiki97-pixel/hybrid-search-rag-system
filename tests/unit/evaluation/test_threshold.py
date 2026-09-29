import json

import pytest

from rag.evaluation.threshold import SweepRow, best_threshold, render, replay, sweep


def r(qtype, conf, correctness=1.0, abstained=False):
    return {"type": qtype, "retrieval_confidence": conf, "correctness": correctness, "abstained": abstained}


RESULTS = [
    r("lookup", 0.95),
    r("lookup", 0.60, correctness=0.5),
    r("multi_hop", 0.40),
    r("no_answer", 0.20, correctness=0.0),  # answered (wrongly) at threshold 0
    r("no_answer", 0.70, abstained=True),  # the model refused even with good retrieval
]


def test_replay_at_zero_keeps_recorded_behaviour():
    row = replay(RESULTS, 0.0)
    assert row.score == pytest.approx((1 + 0.5 + 1 + 0 + 1) / 5)
    assert row.abstain_recall == 0.5 and row.false_abstains == 0


def test_replay_threshold_gates_low_confidence():
    row = replay(RESULTS, 0.3)  # catches the 0.20 no_answer, spares every answerable question
    assert row.score == pytest.approx((1 + 0.5 + 1 + 1 + 1) / 5)
    assert row.abstain_recall == 1.0 and row.false_abstains == 0
    too_high = replay(RESULTS, 0.65)  # now refuses two answerable questions
    assert too_high.false_abstains == 2 and too_high.score == pytest.approx(3 / 5)


def test_missing_confidence_counts_as_zero():
    assert replay([r("lookup", None)], 0.1).false_abstains == 1


def test_best_threshold_takes_middle_of_best_range():
    rows = [SweepRow(t, s, None, 0) for t, s in [(0.0, 0.5), (0.1, 0.9), (0.2, 0.9), (0.3, 0.9), (0.4, 0.7)]]
    assert best_threshold(rows) == 0.2
    assert best_threshold([SweepRow(0.0, 1.0, None, 0)]) == 0.0


def test_sweep_on_example_picks_a_safe_threshold():
    chosen = best_threshold(sweep(RESULTS))
    assert 0.2 < chosen <= 0.4  # between the no_answer at 0.20 and the answerable at 0.40


def test_render_marks_choice():
    rows = sweep(RESULTS, grid=[0.0, 0.3])
    table = render("hybrid", rows, 0.3)
    assert "| 0.30 **<- chosen** | 0.900 | 1.00 | 0 |" in table


def test_sweep_script_writes_report(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("sweep", Path(__file__).parents[3] / "scripts" / "sweep_threshold.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results_file = tmp_path / "hybrid.results.json"
    results_file.write_text(json.dumps({"results": RESULTS}), encoding="utf-8")
    monkeypatch.setattr(module, "OUT", tmp_path / "sweep.md")
    assert module.main([str(results_file)]) == 0
    report = (tmp_path / "sweep.md").read_text(encoding="utf-8")
    assert "## hybrid" in report and "## combined" in report and "<- chosen" in report
    assert module.main([]) == 1
