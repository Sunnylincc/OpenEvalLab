import json

from openevallab.cli import main


def test_cli_demo_command_generates_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_reasoning.jsonl").write_text(
        '{"id":"example_001","task_type":"reasoning","prompt":"2 plus 2?","gold_answer":"4","metadata":{"source":"test"}}\n',
        encoding="utf-8",
    )
    main(["demo"])
    assert (tmp_path / "results" / "demo_results.json").exists()
    assert (tmp_path / "reports" / "demo_report.md").exists()
    payload = json.loads((tmp_path / "results" / "demo_results.json").read_text(encoding="utf-8"))
    assert payload["aggregate_metrics"]["num_examples"] == 1
    assert "demo complete" in capsys.readouterr().out.lower()
