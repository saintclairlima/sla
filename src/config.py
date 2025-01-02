"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = "daphane"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="cleiane-projetos/daphane/vector_store:latest",
    chat_prompt_artifact="cleiane-projetos/daphane/chat_prompt:latest",
    chat_temperature=1.6,
    max_fallback_retries=1,
    model_name="gpt-4o-mini",
    eval_model="gpt-4o-mini",
    eval_artifact="cleiane-projetos/daphane/generated_examples:latest",
)