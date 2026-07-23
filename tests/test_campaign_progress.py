"""Tests for campaign progress reporting."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from galacticsics.campaign.progress import CampaignProgress, stream_ntropy_jsonl


def test_campaign_progress_noop_when_disabled(capsys):
    prog = CampaignProgress(enabled=False)
    prog.banner(name="t", work_root="/tmp", stages=["solve"], n_models=1)
    prog.log("hidden")
    assert capsys.readouterr().out == ""


def test_campaign_progress_logs_when_enabled(capsys):
    prog = CampaignProgress(enabled=True)
    prog.log("hello")
    out = capsys.readouterr().out
    assert "hello" in out
    assert "[" in out  # timestamp


def test_stream_ntropy_jsonl_tail(capsys, tmp_path):
    path = tmp_path / "diagnostics.progress.jsonl"
    stop = threading.Event()

    def writer() -> None:
        for step in range(3):
            rec = {
                "step": step,
                "t_gyr": step * 0.01,
                "n_active": 10,
                "active_fraction": 0.1,
                "dE_over_E0": 0.0,
                "bin_counts": [10],
                "by_type": {},
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
            time.sleep(0.05)
        stop.set()

    threading.Thread(target=writer, daemon=True).start()
    stream_ntropy_jsonl(path, n_steps_total=10, stop_event=stop, poll_s=0.05)
    out = capsys.readouterr().out
    assert "step    0/10" in out
    assert "step    2/10" in out


def test_stream_ntropy_jsonl_recovers_after_truncate(capsys, tmp_path):
    """A truncated progress file must not leave the tailer stuck at a stale offset."""
    path = tmp_path / "diagnostics.progress.jsonl"
    path.write_text("x" * 4096)
    stop = threading.Event()

    def writer() -> None:
        time.sleep(0.05)
        path.write_text("")
        rec = {
            "step": 0,
            "t_gyr": 0.0,
            "n_active": 100,
            "active_fraction": 1.0,
            "dE_over_E0": 0.0,
            "bin_counts": [100],
            "by_type": {},
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
        time.sleep(0.1)
        stop.set()

    threading.Thread(target=writer, daemon=True).start()
    stream_ntropy_jsonl(path, n_steps_total=10, stop_event=stop, poll_s=0.02)
    out = capsys.readouterr().out
    assert "step    0/10" in out
