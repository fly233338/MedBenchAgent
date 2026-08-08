def _task(
    response_format: str = "letter",
    category: str = "image_acquisition_and_basic_information",
) -> dict:
    task = {
        "description": "Identify view",
        "require": "View annotation",
        "task_type": "close_vqa",
        "response_format": response_format,
        "metric": "accuracy",
        "category": category,
        "supporting_fields": ["image.annotation.view"],
        "dataset_support_reason": "Available.",
    }
    return task


def _rag_fields(response_format: str = "letter") -> dict:
    fields = {
        "question_templates": ["Which view is shown?"],
        "generation_logic": "Use the approved supporting fields.",
        "medical_task_description": "This assesses recognition; references support terminology.",
        "citations": [],
    }
    if response_format == "letter":
        fields["options"] = ["CC", "MLO"]
        fields["answer_field"] = "image.annotation.view"
        fields["answer_value_mapping"] = [
            {"raw_value": "CC", "option": "CC"},
            {"raw_value": "MLO", "option": "MLO"},
        ]
    return fields
