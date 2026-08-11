from enum import Enum
from pathlib import Path
import sys
from typing import Any

import pytest

import mnemos_bridge_gemini.adapter as adapter_mod
from mnemos_bridge_gemini import MnemosGeminiAdapter


class FakeMcpClient:
    last_headers = None
    last_timeout = None
    last_token = None

    def __init__(self, *args, **kwargs) -> None:
        # Accept any constructor shape; the offline tests only care that an
        # instance can be returned and list_tools()/call_tool() work.
        pass

    @classmethod
    async def open_from_url(cls, mcp_url, *, token=None, timeout=30):
        # mnemos-bridge-core v0.1.2+ entry point — what the adapter prefers.
        cls.last_url = mcp_url
        cls.last_token = token
        cls.last_timeout = timeout
        # Backward-compat: synthesize the headers shape the test still asserts.
        cls.last_headers = {"Authorization": f"Bearer {token}"} if token else None
        return cls()

    @classmethod
    def from_url(cls, mcp_url, *, token=None, timeout=30):
        # Synchronous factory used by the older fallback path.
        cls.last_url = mcp_url
        cls.last_token = token
        cls.last_timeout = timeout
        cls.last_headers = {"Authorization": f"Bearer {token}"} if token else None
        return cls()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    @classmethod
    async def connect(cls, mcp_url, *, headers=None, timeout=30):
        cls.last_url = mcp_url
        cls.last_headers = headers
        cls.last_timeout = timeout
        return cls()

    async def list_tools(self):
        return [
            {
                "name": "search_memories",
                "description": "Search MNEMOS memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "filters": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "scope": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_memory",
                "description": "Read one memory by id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
            },
            {
                "name": "tag_memory",
                "description": "Tag a memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tag": {"$ref": "#/$defs/tag"},
                    },
                    "$defs": {
                        "tag": {
                            "type": "string",
                            "deprecated": True,
                        }
                    },
                    "required": ["tag"],
                },
            },
        ]


def _function_declarations(tools):
    assert isinstance(tools, list)
    assert tools
    tool = tools[0]
    if isinstance(tool, dict):
        return tool["function_declarations"]
    return list(tool.function_declarations)


def _declaration_to_dict(declaration):
    if isinstance(declaration, dict):
        return declaration
    return {
        "name": declaration.name,
        "description": declaration.description,
        "parameters": _to_plain_dict(declaration.parameters),
    }


def _to_plain_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: _to_plain_dict(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(child) for child in value]
    if isinstance(value, Enum):
        return value.value.lower()
    return value


@pytest.mark.asyncio
async def test_gemini_tools_returns_function_declarations_and_strips_unsupported_keywords(
    monkeypatch,
):
    monkeypatch.setattr(adapter_mod, "McpClient", FakeMcpClient)
    monkeypatch.setattr(adapter_mod, "SchemaTranslator", None)

    adapter = await MnemosGeminiAdapter.connect(
        "http://127.0.0.1:5003",
        "test-token",
        timeout=12,
    )
    tools = await adapter.gemini_tools()
    declarations = [_declaration_to_dict(item) for item in _function_declarations(tools)]

    search = declarations[0]
    assert search["name"] == "search_memories"
    assert search["description"] == "Search MNEMOS memories."
    assert search["parameters"]["required"] == ["query"]
    assert "additionalProperties" not in search["parameters"]
    assert "additionalProperties" not in search["parameters"]["properties"]["filters"]
    assert "oneOf" not in search["parameters"]["properties"]["scope"]


@pytest.mark.asyncio
async def test_connect_passes_bearer_header_and_required_arrays_are_preserved(monkeypatch):
    monkeypatch.setattr(adapter_mod, "McpClient", FakeMcpClient)
    monkeypatch.setattr(adapter_mod, "SchemaTranslator", None)

    adapter = await MnemosGeminiAdapter.connect(
        "http://mnemos.example",
        "secret-token",
        timeout=7,
    )
    tools = await adapter.gemini_tools()
    declarations = [_declaration_to_dict(item) for item in _function_declarations(tools)]

    assert FakeMcpClient.last_headers == {"Authorization": "Bearer secret-token"}
    assert FakeMcpClient.last_timeout == 7
    assert declarations[1]["parameters"]["required"] == ["memory_id"]
    assert declarations[2]["parameters"]["required"] == ["tag"]
    assert declarations[2]["parameters"]["properties"]["tag"] == {"type": "string"}
    assert "$defs" not in declarations[2]["parameters"]


@pytest.mark.asyncio
async def test_connect_uses_default_timeout(monkeypatch):
    monkeypatch.setattr(adapter_mod, "McpClient", FakeMcpClient)
    monkeypatch.setattr(adapter_mod, "SchemaTranslator", None)

    await MnemosGeminiAdapter.connect("http://mnemos.example", "secret-token")

    assert FakeMcpClient.last_timeout == 30


def test_core_translation_receives_dereferenced_schema(monkeypatch):
    core_src = Path(__file__).resolve().parents[2] / "mnemos-bridge-core" / "src"
    if not core_src.exists():
        pytest.skip("adjacent mnemos-bridge-core checkout is unavailable")
    sys.path.insert(0, str(core_src))
    from mnemos_bridge_core import SchemaTranslator as CoreSchemaTranslator
    from mnemos_bridge_core import ToolSchema

    source_schema = {
        "type": "object",
        "properties": {"tag": {"$ref": "#/$defs/tag"}},
        "$defs": {"tag": {"type": "string"}},
        "required": ["tag"],
    }
    monkeypatch.setattr(adapter_mod, "SchemaTranslator", CoreSchemaTranslator)

    translated = adapter_mod._translate_parameters(
        ToolSchema(
            name="tag_memory",
            description="Tag a memory.",
            input_schema=source_schema,
        )
    )

    assert translated["properties"]["tag"] == {"type": "string"}
    assert "$defs" not in translated


def test_translation_preserves_gemini_supported_constraints_with_core_installed(monkeypatch):
    core_src = Path(__file__).resolve().parents[2] / "mnemos-bridge-core" / "src"
    if not core_src.exists():
        pytest.skip("adjacent mnemos-bridge-core checkout is unavailable")
    sys.path.insert(0, str(core_src))
    from mnemos_bridge_core import SchemaTranslator as CoreSchemaTranslator
    from mnemos_bridge_core import ToolSchema

    monkeypatch.setattr(adapter_mod, "SchemaTranslator", CoreSchemaTranslator)
    translated = adapter_mod._translate_parameters(
        ToolSchema(
            name="list_memories",
            description="List memories.",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "default": 20,
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 50},
                        "maxItems": 10,
                    },
                },
            },
        )
    )

    assert translated["properties"]["limit"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 500,
        "default": 20,
    }
    assert translated["properties"]["tags"] == {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 50},
        "maxItems": 10,
    }


@pytest.mark.parametrize(
    "tool",
    [
        {
            "name": "search_memories",
            "description": "Search memories.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        type(
            "OfficialMcpToolShape",
            (),
            {
                "name": "search_memories",
                "description": "Search memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        )(),
    ],
    ids=["mapping", "mcp-inputSchema-object"],
)
def test_core_translation_normalizes_public_duck_typed_tools(monkeypatch, tool):
    core_src = Path(__file__).resolve().parents[2] / "mnemos-bridge-core" / "src"
    if not core_src.exists():
        pytest.skip("adjacent mnemos-bridge-core checkout is unavailable")
    sys.path.insert(0, str(core_src))
    from mnemos_bridge_core import SchemaTranslator as CoreSchemaTranslator
    from mnemos_bridge_core import ToolSchema as CoreToolSchema

    monkeypatch.setattr(adapter_mod, "SchemaTranslator", CoreSchemaTranslator)
    monkeypatch.setattr(adapter_mod, "ToolSchema", CoreToolSchema)

    translated = adapter_mod._translate_parameters(tool)

    assert translated == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


def test_documented_mnemos_endpoints_include_sse_route():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    assert readme.count("http://192.168.207.67:5003/sse") == 3
    assert "http://192.168.207.67:5003\"" not in readme
    assert "http://192.168.207.67:5003\n" not in readme


@pytest.mark.asyncio
async def test_aclose_falls_back_to_async_context_exit():
    class ContextManagedClient:
        def __init__(self):
            self.exit_args = None

        async def __aexit__(self, *args):
            self.exit_args = args

    client = ContextManagedClient()
    adapter = MnemosGeminiAdapter(client)

    await adapter.aclose()

    assert client.exit_args == (None, None, None)


@pytest.mark.asyncio
async def test_connect_closes_client_when_initial_tool_discovery_fails(monkeypatch):
    discovery_error = RuntimeError("tool discovery failed")

    class FailingClient:
        def __init__(self):
            self.closed = False

        async def list_tools(self):
            raise discovery_error

        async def aclose(self):
            self.closed = True

    client = FailingClient()

    async def connect_client(*args, **kwargs):
        return client

    monkeypatch.setattr(adapter_mod, "McpClient", object())
    monkeypatch.setattr(adapter_mod, "_connect_mcp_client", connect_client)

    with pytest.raises(RuntimeError) as caught:
        await MnemosGeminiAdapter.connect("http://mnemos.example", "secret-token")

    assert caught.value is discovery_error
    assert client.closed is True
