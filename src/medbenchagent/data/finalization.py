"""Canonical prepared-dataset publication boundary."""

from __future__ import annotations

from medbenchagent.preparation.publisher import DatasetPublisher


def finalize_prepared_dataset(candidate, context):
    return DatasetPublisher(context["bundle_root"]).publish(candidate, context)
