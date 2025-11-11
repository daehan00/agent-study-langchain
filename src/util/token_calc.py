from typing import Any
from langchain.messages import AnyMessage, AIMessage

gemini_base = 1_000_000

PRICES = {
    "gemini-2.5-flash-lite": {
        "input": (0.1 / gemini_base, 0.1 / gemini_base),
        "output": (0.4 / gemini_base, 0.4 / gemini_base)
    },
    "gemini-2.5-flash": {
        "input": (0.3 / gemini_base, 0.3 / gemini_base),
        "output": (2.5 / gemini_base, 2.5 / gemini_base)
    },
    "gemini-2.5-pro": {
        "input": (1.25 / gemini_base, 2.5 / gemini_base),
        "output": (10 / gemini_base, 15 / gemini_base)
    }
}

def _check_valid_message(message: AIMessage) -> tuple[bool, str]:
    if not message.usage_metadata:
        return False, "missing key: usage_metadata"
    
    required_keys = ["input_tokens", "output_tokens"]
    for key in required_keys:
        if key not in message.usage_metadata:
            return False, f"missing key: {key}. available keys: {list(message.usage_metadata.keys())}"
    
    return True, ""

def _check_and_load_model_name(model_name: str) -> str | None:
    model = None
    for key in PRICES.keys():
        if model_name.startswith(key):
            model = key
    return model

def _calculate_price(model: str, input_tokens: float, output_tokens: float) -> tuple[float, float]:
    input_price_per = PRICES[model]["input"][0] if input_tokens <= 200_000 else PRICES[model]["input"][1]
    output_price_per = PRICES[model]["output"][0] if output_tokens <= 200_000 else PRICES[model]["output"][1]

    return input_tokens * input_price_per, output_tokens * output_price_per

def _log_cost(input_cost: float, output_cost: float, multi: bool = False) -> str:
    message = f" 입력 토큰 비용: {input_cost:.5f}, 출력 토큰 비용: {output_cost:.5f}, 총비용: {input_cost+output_cost:.5f}"
    if multi:
        message = "[다중 메시지]" + message
    else:
        message = "[단일 메시지]" + message
    
    print(message)
    return message


def calculate_price_of_message(message: AIMessage, debug: bool = False) -> tuple[float, float]:
    """
    AIMessage를 입력으로 받아서 입력 비용, 출력 비용을 리턴하는 함수.

    Args:
        message: AnyMessage
        debug: bool = False
    Return:
        tuple[float, float] = input_price, output_price
    """
    valid, msg = _check_valid_message(message)
    if not valid:
        raise KeyError(msg)

    model_name = message.response_metadata["model_name"]
    model = _check_and_load_model_name(model_name)
    if not model:
        raise KeyError(f"{model_name} : 정의된 모델이름이 아닙니다!")

    input_tokens = message.usage_metadata["input_tokens"] # type: ignore
    output_tokens = message.usage_metadata["output_tokens"] # type: ignore
    input_p, output_p = _calculate_price(model, input_tokens, output_tokens)

    if debug:
        _log_cost(input_p, output_p)
    return input_p, output_p


def calculate_price_of_messages(messages: list[AnyMessage], debug: bool = False) -> Any:
    results = [0.0, 0.0]
    
    for i, message in enumerate(messages, 1):
        if type(message) is not AIMessage:
            continue

        # if debug:
            # print(f"[{i}번째 메시지] (단위: $)")

        input_p, output_p = calculate_price_of_message(message, debug)

        results[0] += input_p
        results[1] += output_p
    
    return tuple(results)