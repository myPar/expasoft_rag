from llama_index.llms.openrouter import OpenRouter
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper


# OpenRouter wrapper for llamaindex
def create_llm(config: dict, llm_name: str):
    token = config['token']
    llm_config = config[llm_name]

    open_router_llm = OpenRouter(
        api_key = token,
        max_tokens = llm_config['max_tokens'],
        context_window = llm_config['context'],
        model = llm_config['model'],
        max_retries=llm_config['max_retries'],
        timeout=llm_config['response_timeout'],
        additional_kwargs = {
            "extra_body": {
                "provider": llm_config['provider']
            }
        }
    )

    return open_router_llm


# Langchain wrapper upon OpenAI
def create_judge_llm(config: dict, llm_name:str):
    token = config['token']
    llm_config = config[llm_name]

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=llm_config['model'],
            openai_api_key=token,
            openai_api_base="https://openrouter.ai/api/v1",  # important!
            max_completion_tokens=llm_config['max_tokens'],
            max_retries=llm_config['max_retries'],
            timeout=llm_config['response_timeout'],

            extra_body={
                "provider": llm_config['provider']
            }
        )
    )

    return evaluator_llm