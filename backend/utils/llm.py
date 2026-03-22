import logging
from functools import lru_cache
from typing import Any

import boto3
from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> Any:
    provider = settings.model_provider.strip().lower()

    if provider == "gemini":
        logger.info("Initializing Gemini model: %s", settings.gemini_model)
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.model_temperature,
            max_output_tokens=settings.model_max_tokens,
        )

    logger.info("Initializing Bedrock model: %s", settings.bedrock_model_id)
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    return ChatBedrock(
        client=bedrock_client,
        model_id=settings.bedrock_model_id,
        model_kwargs={
            "temperature": settings.model_temperature,
            "max_tokens": settings.model_max_tokens,
        },
    )
