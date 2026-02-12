import json
from llama_index.llms.openrouter import OpenRouter


def create_llm(config_path:str, llm_name: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
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
    