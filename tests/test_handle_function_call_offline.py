from dataclasses import dataclass

import pytest

import mnemos_bridge_gemini.adapter as adapter_mod
from mnemos_bridge_gemini import MnemosGeminiAdapter


@dataclass
class FunctionCallMock:
    name: str
    args: dict


class FakeMcpClient:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"result for {name}: {args}",
                }
            ]
        }


@pytest.mark.asyncio
async def test_handle_function_call_accepts_dict_input(monkeypatch):
    monkeypatch.setattr(adapter_mod, "ResultRenderer", None)
    client = FakeMcpClient()
    adapter = MnemosGeminiAdapter(client)

    response = await adapter.handle_function_call(
        {"name": "search_memories", "args": {"query": "infrastructure"}}
    )

    assert client.calls == [("search_memories", {"query": "infrastructure"})]
    assert response.function_response.name == "search_memories"
    assert response.function_response.response["result"]
    assert "infrastructure" in response.function_response.response["result"]


@pytest.mark.asyncio
async def test_handle_function_call_accepts_dataclass_style_input(monkeypatch):
    monkeypatch.setattr(adapter_mod, "ResultRenderer", None)
    client = FakeMcpClient()
    adapter = MnemosGeminiAdapter(client)

    response = await adapter.handle_function_call(
        FunctionCallMock(name="read_memory", args={"memory_id": "mem_123"})
    )

    assert client.calls == [("read_memory", {"memory_id": "mem_123"})]
    assert response.function_response.name == "read_memory"
    assert "result" in response.function_response.response
    assert isinstance(response.function_response.response["result"], str)
    assert response.function_response.response["result"]


@pytest.mark.asyncio
async def test_handle_function_call_preserves_mcp_error_semantics(monkeypatch):
    monkeypatch.setattr(adapter_mod, "ResultRenderer", None)

    class ErrorClient:
        async def call_tool(self, name, args):
            return {"content": ["unauthorized"], "is_error": True}

    adapter = MnemosGeminiAdapter(ErrorClient())
    response = await adapter.handle_function_call({"name": "read_memory", "args": {}})

    assert response.function_response.response == {
        "error": "unauthorized",
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_handle_function_call_extracts_core_renderer_content(monkeypatch):
    class CoreRenderer:
        @staticmethod
        def to_gemini_part(result):
            return {
                "functionResponse": {
                    "response": {"content": ["memory found", "second line"]}
                }
            }

    monkeypatch.setattr(adapter_mod, "ResultRenderer", CoreRenderer)
    client = FakeMcpClient()
    adapter = MnemosGeminiAdapter(client)

    response = await adapter.handle_function_call({"name": "read_memory", "args": {}})

    assert response.function_response.response == {
        "result": "memory found\nsecond line"
    }
