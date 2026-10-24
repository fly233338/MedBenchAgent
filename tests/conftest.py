"""Tests use explicit fake models and local fixtures; no automatic remote calls."""

import os

# SDK imports must remain offline and use the packaged cost metadata.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
