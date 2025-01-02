"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = "daphane"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="51ai/daphane/vector_store:latest",
    chat_prompt_artifact="51ai/daphane/chat_prompt:latest",
    chat_temperature=1.6,
    max_fallback_retries=1,
    model_name="llama3.1",
    eval_model="llama3.1",
    eval_artifact="51ai/daphane/generated_examples:latest",
)