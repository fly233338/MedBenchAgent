from medbenchagent.evaluation.config import EvaluationConfig, MetricSpec
from medbenchagent.planning.task_space import CapabilityCategory, OutputSpec, TaskParadigm


def test_metric_selection_is_post_planning_and_task_specific():
    config = EvaluationConfig(metrics=[
        MetricSpec(name="exact", output_spec="label"),
        MetricSpec(name="bleu", output_spec="text", task_paradigm="Report Generation"),
    ])
    selected = config.for_task(output_spec=OutputSpec.TEXT, task_paradigm=TaskParadigm.REPORT_GENERATION,
                               capability_category=CapabilityCategory.GENERATION)
    assert [metric.name for metric in selected] == ["bleu"]

