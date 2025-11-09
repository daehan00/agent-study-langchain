from typing import List, Any, Optional
import uuid

"""LangGraph 디버깅용 메시지 출력 헬퍼 함수"""



def print_messages(
    messages: List[Any],
    mode: str = "full",
    show_metadata: bool = True,
    show_content: bool = True,
    max_content_length: Optional[int] = None
) -> None:
    """
    LangGraph response["messages"]를 예쁘게 출력하는 함수
    
    Args:
        messages: LangGraph의 메시지 리스트
        mode: 출력 모드 ("full", "summary", "metadata", "content")
        show_metadata: 메타데이터 표시 여부
        show_content: 콘텐츠 표시 여부
        max_content_length: 콘텐츠 최대 길이 (None이면 전체 출력)
    """
    print("\n" + "="*80)
    print(f"📨 총 {len(messages)}개의 메시지")
    print("="*80 + "\n")
    
    for idx, msg in enumerate(messages, 1):
        if mode == "summary":
            _print_summary(msg, idx)
        elif mode == "metadata":
            _print_metadata(msg, idx)
        elif mode == "content":
            _print_content(msg, idx, max_content_length)
        else:  # full
            _print_full(msg, idx, show_metadata, show_content, max_content_length)
        
        print("-" * 80 + "\n")


def _print_summary(msg: Any, idx: int) -> None:
    """요약 정보만 출력"""
    msg_type = getattr(msg, "type", "unknown")
    role = getattr(msg, "role", getattr(msg.__class__, "__name__", "unknown"))
    content_preview = _get_content_preview(msg, 50)
    
    print(f"[{idx}] {role.upper()} ({msg_type})")
    print(f"    💬 {content_preview}")


def _print_metadata(msg: Any, idx: int) -> None:
    """메타데이터만 출력"""
    msg_type = getattr(msg, "type", "unknown")
    role = getattr(msg, "role", getattr(msg.__class__, "__name__", "unknown"))
    
    print(f"[{idx}] 메타데이터")
    print(f"    Type: {msg_type}")
    print(f"    Role: {role}")
    
    # ID 정보
    if hasattr(msg, "id"):
        print(f"    ID: {msg.id}")
    
    # 추가 메타데이터
    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
        print(f"    Additional kwargs: {msg.additional_kwargs}")
    
    if hasattr(msg, "response_metadata") and msg.response_metadata:
        print(f"    Response metadata: {msg.response_metadata}")
    
    # 토큰 사용량
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        print(f"    Usage: {msg.usage_metadata}")


def _print_content(msg: Any, idx: int, max_length: Optional[int] = 100) -> None:
    """콘텐츠만 출력"""
    role = getattr(msg, "role", getattr(msg.__class__, "__name__", "unknown"))
    content = _get_content(msg)
    
    if max_length and len(str(content)) > max_length:
        content = str(content)[:max_length] + "..."
    
    print(f"[{idx}] {role.upper()}")
    print(f"{content}")


def _print_full(
    msg: Any,
    idx: int,
    show_metadata: bool,
    show_content: bool,
    max_length: Optional[int] = 100
) -> None:
    """전체 정보 출력 (JSON 형태)"""
    import json
    
    msg_type = getattr(msg, "type", "unknown")
    role = getattr(msg, "role", getattr(msg.__class__, "__name__", "unknown"))
    
    print(
        f"""index: {idx},
type": {msg_type.upper()}"""
    )


    output = {}
    
    if show_content:
        content = _get_content(msg)
        if max_length and len(str(content)) > max_length:
            content = str(content)[:max_length] + "..."
        output["content"] = content
    
    if show_metadata:
        # 객체의 모든 속성을 딕셔너리로 변환
        all_attributes = _obj_to_dict(msg, max_depth=5)
        
        # __class__ 키는 제거 (이미 type과 role로 표현됨)
        if isinstance(all_attributes, dict) and "__class__" in all_attributes:
            del all_attributes["__class__"]
        
        output["metadata"] = all_attributes
    
    print(json.dumps(output, indent=3, ensure_ascii=False))


def _get_content(msg: Any) -> str:
    """메시지에서 콘텐츠 추출"""
    if hasattr(msg, "content"):
        return str(msg.content)
    return str(msg)


def _get_content_preview(msg: Any, max_length: int = 100) -> str:
    """콘텐츠 미리보기 생성"""
    content = _get_content(msg)
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content

def print_tool_invoke_info(
    tool_name: str,
    tool_call_id: str,
    args: dict
) -> None:
    """
    Tool invoke 시작 정보를 출력
    
    Args:
        tool_name: tool 이름
        tool_call_id: tool call ID
        args: tool 인자
    """
    print("\n" + "="*80)
    print("🔧 Tool Invoke 테스트")
    print("="*80)
    print(f"📛 Tool Name: {tool_name}")
    print(f"🔑 Call ID: {tool_call_id}")
    print(f"\n📋 Input Arguments:")
    for key, value in args.items():
        print(f"   - {key}: {value}")
    print("\n⏳ Invoking tool...")


def _obj_to_dict(obj: Any, max_depth: int = 3, current_depth: int = 0) -> Any:
    """
    객체를 딕셔너리로 변환 (JSON 직렬화 가능하도록)
    
    Args:
        obj: 변환할 객체
        max_depth: 최대 재귀 깊이
        current_depth: 현재 깊이
    
    Returns:
        직렬화 가능한 객체
    """
    if current_depth > max_depth:
        return str(obj)
    
    # 기본 타입
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # 딕셔너리
    if isinstance(obj, dict):
        return {k: _obj_to_dict(v, max_depth, current_depth + 1) for k, v in obj.items()}
    
    # 리스트, 튜플
    if isinstance(obj, (list, tuple)):
        return [_obj_to_dict(item, max_depth, current_depth + 1) for item in obj]
    
    # 클래스 객체 (Pydantic 모델 등)
    if hasattr(obj, '__dict__'):
        return {
            '__class__': obj.__class__.__name__,
            **{k: _obj_to_dict(v, max_depth, current_depth + 1) for k, v in obj.__dict__.items()}
        }
    
    # 변환 불가능한 경우 문자열로
    return str(obj)


def print_command_result(command: Any, max_content_length: Optional[int] = 500) -> None:
    """
    Command 객체를 JSON 형식으로 출력 (하위 호환성 유지)
    
    Args:
        command: Command 객체
        max_content_length: 콘텐츠 최대 길이 (각 필드별)
    """
    print_json_result(command, title="Command 객체", max_content_length=max_content_length)


def _truncate_text(text: str, max_length: int) -> str:
    """텍스트를 지정된 길이로 자르기"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def print_json_result(obj: Any, title: str = "객체", max_content_length: Optional[int] = 500) -> None:
    """
    객체를 JSON 형식으로 출력 (Command, ToolMessage 등)
    
    Args:
        obj: 출력할 객체
        title: 출력 제목
        max_content_length: 콘텐츠 최대 길이 (각 필드별)
    """
    import json
    
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")
    
    obj_dict = _obj_to_dict(obj)
    
    json_str = json.dumps(obj_dict, indent=2, ensure_ascii=False)
    
    if max_content_length:
        lines = json_str.split('\n')
        truncated_lines = []
        for line in lines:
            if len(line) > max_content_length:
                if line.rstrip().endswith((',', '{', '[', '}', ']')):
                    last_char = line.rstrip()[-1]
                    truncated_lines.append(line[:max_content_length] + '..." ' + last_char)
                else:
                    truncated_lines.append(line[:max_content_length] + '..."')
            else:
                truncated_lines.append(line)
        json_str = '\n'.join(truncated_lines)
    
    print(json_str)
    print("\n" + "="*80 + "\n")


def print_tool_invoke_result(result: Any, success: bool = True, error: Optional[str] = None) -> None:
    """
    Tool invoke 결과를 출력
    
    Args:
        result: tool 실행 결과
        success: 성공 여부
        error: 에러 메시지 (실패 시)
    """
    if success:
        print("\n✅ Tool Execution Success")
        print(f"\n📤 Result:")
        
        if hasattr(result, 'update') or result.__class__.__name__ == 'Command':
            print_json_result(result, title="Command 객체")
            return
        
        if result.__class__.__name__ in ['ToolMessage', 'AIMessage', 'HumanMessage', 'SystemMessage']:
            print_json_result(result, title=f"{result.__class__.__name__} 객체")
            return
        
        if hasattr(result, '__dict__'):
            print_json_result(result, title=f"{result.__class__.__name__} 객체")
            return
        
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"   - {key}: {_truncate_text(str(value), 200)}")
        
        elif isinstance(result, list):
            print(f"   List with {len(result)} items:")
            for idx, item in enumerate(result):
                if hasattr(item, '__class__') and item.__class__.__name__ in ['ToolMessage', 'AIMessage', 'HumanMessage', 'SystemMessage', 'Command']:
                    print(f"\n   [{idx}]")
                    print_json_result(item, title=f"{item.__class__.__name__}")
                else:
                    print(f"   [{idx}] {_truncate_text(str(item), 200)}")
        
        else:
            print(f"   {_truncate_text(str(result), 200)}")
    else:
        print(f"\n❌ Tool Execution Failed")
        print(f"Error: {error}")
    print("="*80 + "\n")


def test_tool_invoke(
    tool: Any,
    args: dict,
    tool_call_id: Optional[str] = None,
    print_result: bool = True
) -> Any:
    """
    Tool을 직접 invoke하고 결과를 출력하는 테스트 함수
    
    Args:
        tool: 테스트할 tool 객체 (invoke 메서드를 가진 객체)
        args: tool에 전달할 인자 딕셔너리
        tool_call_id: tool call ID (None이면 자동 생성)
        print_result: 결과 출력 여부
    
    Returns:
        tool 실행 결과
    
    Example:
        >>> from langchain_core.tools import tool
        >>> @tool
        ... def search(query: str) -> str:
        ...     return f"Searching for: {query}"
        >>> result = test_tool_invoke(search, {"query": "Python"})
    """
    if tool_call_id is None:
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
    
    tool_name = getattr(tool, "name", tool.__class__.__name__)
    
    if print_result:
        print_tool_invoke_info(tool_name, tool_call_id, args)
    
    try:
        result = tool.invoke({
            "args": args,
            "name": tool_name,
            "type": "tool_call",
            "id": tool_call_id
        })
        
        if print_result:
            print_tool_invoke_result(result, success=True)
        
        return result
    
    except Exception as e:
        if print_result:
            print_tool_invoke_result(None, success=False, error=str(e))
        raise

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

def calculate_price_by_token(message: dict[str, Any]) -> Any:
    input_tokens = message["usage_metadata"]["input_tokens"]
    output_tokens = message["usage_metadata"]["output_tokens"]
    model_name = message["response_metadata"]["model_name"]

    model = ""
    for key in PRICES.keys():
        if model_name.startswith(key):
            model = key

    if not model:
        raise KeyError(f"{model_name} : 정의된 모델이름이 아닙니다!")
    
    try:
        input_price_per = PRICES[model]["input"][0] if input_tokens <= 200_000 else PRICES[model]["input"][1]
        output_price_per = PRICES[model]["output"][0] if output_tokens <= 200_000 else PRICES[model]["output"][1]

        return input_tokens * input_price_per, output_tokens * output_price_per
    except Exception as e:
        print(e)



# 사용 예시
if __name__ == "__main__":
    # 예시 사용법
    print("사용 예시:")
    print("1. 전체 출력: print_messages(response['messages'])")
    print("2. 요약만: print_messages(response['messages'], mode='summary')")
    print("3. 메타데이터만: print_messages(response['messages'], mode='metadata')")
    print("4. 콘텐츠만: print_messages(response['messages'], mode='content')")
    print("5. 콘텐츠 길이 제한: print_messages(response['messages'], max_content_length=100)")
    print("6. Tool invoke 테스트: test_tool_invoke(tool, args={}, print_result=True)")
    