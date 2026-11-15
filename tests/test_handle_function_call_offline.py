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
async def test_handle_function_call_preserves_function_call_id(monkeypatch):
    monkeypatch.setattr(adapter_mod, "ResultRenderer", None)
    adapter = MnemosGeminiAdapter(FakeMcpClient())

    response = await adapter.handle_function_call(
        {
            "id": "call-42",
            "name": "search_memories",
            "args": {"query": "infrastructure"},
        }
    )

    assert response.function_response.id == "call-42"


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
async def test_explicit_tool_allowlist_rejects_undeclared_tool(monkeypatch):
    monkeypatch.setattr(adapter_mod, "ResultRenderer", None)
    client = FakeMcpClient()
    adapter = MnemosGeminiAdapter(client, tools=[{"name": "read_memory"}])

    with pytest.raises(ValueError, match="not in this adapter's declared tool set"):
        await adapter.handle_function_call({"name": "delete_memory", "args": {}})

    assert client.calls == []


@pytest.mark.asyncio
async def test_connect_tool_allowlist_rejects_undeclared_tool(monkeypatch):
    class ConnectedClient(FakeMcpClient):
        async def list_tools(self):
            return [{"name": "read_memory"}]

    client = ConnectedClient()

    async def connect_client(*args, **kwargs):
        return client

    monkeypatch.setattr(adapter_mod, "McpClient", object())
    monkeypatch.setattr(adapter_mod, "_connect_mcp_client", connect_client)

    adapter = await MnemosGeminiAdapter.connect(
        "http://mnemos.example",
        "secret-token",
    )

    with pytest.raises(ValueError, match="not in this adapter's declared tool set"):
        await adapter.handle_function_call({"name": "delete_memory", "args": {}})

    assert client.calls == []


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


class McpError(Exception):
    """Test stand-in for the core/MCP transport error type."""


@pytest.mark.asyncio
async def test_handle_function_call_returns_error_part_on_mcp_failure(monkeypatch):
    monkeypatch.setattr(adapter_mod, "ResultRenderer", None)

    class FailingClient:
        async def call_tool(self, name, args):
            raise McpError("boom: connection reset")

    adapter = MnemosGeminiAdapter(FailingClient())
    response = await adapter.handle_function_call(
        {"id": "call-7", "name": "read_memory", "args": {}}
    )

    assert response.function_response.id == "call-7"
    assert response.function_response.name == "read_memory"
    payload = response.function_response.response
    assert payload["is_error"] is True
    assert "boom: connection reset" in payload["error"]


@pytest.mark.asyncio
async def test_handle_function_call_returns_error_part_on_timeout(monkeypatch):
    import asyncio

    monkeypatch.setattr(adapter_mod, "ResultRenderer", None)

    class TimeoutClient:
        async def call_tool(self, name, args):
            raise asyncio.TimeoutError("timed out")

    adapter = MnemosGeminiAdapter(TimeoutClient())
    response = await adapter.handle_function_call({"name": "read_memory", "args": {}})

    assert response.function_response.response["is_error"] is True
    assert "timed out" in response.function_response.response["error"]
