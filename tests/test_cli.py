from medbenchagent.cli import _parser


def test_parser_exposes_terminal_commands():
    parser = _parser()
    args = parser.parse_args(["run", "--source-manifest", "sources.json", "--goal-kind", "plan", "--plan-level", "scenarios"])
    assert args.goal_kind == "plan" and args.plan_level == "scenarios"
    assert parser.parse_args(["resume", "--directory", "saved", "--answer", "yes"]).answer == "yes"
    assert parser.parse_args(["register", "--dataset-id", "example", "--manifest", "dataset.json"]).kind == "dataset"
