import json
from llama_index.llms.openrouter import OpenRouter


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
    