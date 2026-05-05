"""Google Gemini adapter for MNEMOS MCP tools."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from typing import Any, Mapping

from google import genai
from google.genai import types

try:  # pragma: no cover - exercised only when mnemos-bridge-core is installed.
    from mnemos_bridge_core import McpClient, ResultRenderer, SchemaTranslator
except ImportError:  # pragma: no cover - offline tests patch these symbols.
    McpClient = None  # type: ignore[assignment]
    ResultRenderer = None  # type: ignore[assignment]
    SchemaTranslator = None  # type: ignore[assignment]

_GENAI_CLIENT_TYPE = genai.Client


UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "$defs",
    "additionalProperties",
    "allOf",
    "anyOf",
    "contentEncoding",
    "contentMediaType",
    "definitions",
    "deprecated",
    "else",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "readOnly",
    "then",
    "writeOnly",
}


class MnemosGeminiAdapter:
    """Adapt MNEMOS MCP tools to Google Gemini function calling."""

    def __init__(self, client: Any, tools: list[Any] | None = None) -> None:
        self._client = client
        self._tools = list(tools or [])

    @classmethod
    async def connect(
        cls,
        mcp_url: str,
        mcp_token: str,
        *,
        timeout: float = 30,
    ) -> "MnemosGeminiAdapter":
        """Connect to MNEMOS MCP HTTP/SSE endpoint and fetch available tools.

        TODO: This assumes ``mnemos_bridge_core.McpClient`` accepts either an async
        ``connect(url, headers=..., timeout=...)`` factory or constructor arguments with
        the same values. Adjust the call path if the core package exposes a narrower API.
        """

        if McpClient is None:
            raise RuntimeError(
                "mnemos_bridge_core is required to connect. Install mnemos-bridge-core>=0.1.0."
            )

        headers = {"Authorization": f"Bearer {mcp_token}"}
        client = await _connect_mcp_client(mcp_url, headers=headers, timeout=timeout)
        tools = await _fetch_tools(client)
        return cls(client, tools)

    async def gemini_tools(self) -> list[types.Tool]:
        """Return Gemini Tool[] list with translated function declarations."""

        if not self._tools:
            self._tools = await _fetch_tools(self._client)

        function_declarations = [_function_declaration_from_tool(tool) for tool in self._tools]
        if not function_declarations:
            return []

        return [types.Tool(function_declarations=function_declarations)]

    async def handle_function_call(self, fc: Any) -> types.Part:
        """Dispatch a Gemini FunctionCall to MCP and return a response part payload."""

        name = _get_value(fc, "name")
        if not name:
            raise ValueError("Gemini function call is missing a name.")

        args = _coerce_args(_get_value(fc, "args", {}))
        result = await _maybe_await(self._client.call_tool(name, args))
        rendered = await _render_result_for_gemini(result)

        return types.Part(
            function_response=types.FunctionResponse(
                name=name,
                response={"result": rendered},
            )
        )

    async def aclose(self) -> None:
        """Close the underlying MCP client connection."""

        for method_name in ("aclose", "close", "disconnect"):
            method = getattr(self._client, method_name, None)
            if method is not None:
                await _maybe_await(method())
                return

    async def __aenter__(self) -> "MnemosGeminiAdapter":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()


async def _connect_mcp_client(mcp_url: str, *, headers: dict[str, str], timeout: float) -> Any:
    # Strip 'Bearer ' prefix if present — core's McpClient takes the raw token.
    auth_value = headers.get("Authorization", "")
    token = auth_value.removeprefix("Bearer ").removeprefix("bearer ") if auth_value else ""

    # Preferred path (mnemos-bridge-core v0.1.2+): open_from_url constructs +
    # opens the SSE session in one call. Falls back to from_url + __aenter__
    # for older core builds.
    opener = getattr(McpClient, "open_from_url", None)
    if opener is not None:
        return await opener(mcp_url, token=token, timeout=timeout)

    from_url = getattr(McpClient, "from_url", None)
    if from_url is not None:
        client = from_url(mcp_url, token=token, timeout=timeout)
        aenter = getattr(client, "__aenter__", None)
        if aenter is not None:
            await aenter()
        return client

    connector = getattr(McpClient, "connect", None)
    if connector is not None:
        return await connector(mcp_url, headers=headers, timeout=timeout)

    # Last-resort: direct constructor + open. Core's McpClient signature is
    # __init__(url, *, token, timeout).
    client = McpClient(mcp_url, token=token, timeout=timeout)
    aenter = getattr(client, "__aenter__", None)
    if aenter is not None:
        await aenter()
    return client


async def _fetch_tools(client: Any) -> list[Any]:
    list_tools = getattr(client, "list_tools", None)
    if list_tools is not None:
        response = await _maybe_await(list_tools())
    else:
        response = getattr(client, "tools", [])

    if isinstance(response, Mapping):
        response = response.get("tools", [])
    else:
        response = _get_value(response, "tools", response)

    return list(response or [])


def _function_declaration_from_tool(tool: Any) -> types.FunctionDeclaration:
    name = _get_value(tool, "name")
    description = _get_value(tool, "description", "") or ""
    parameters = _translate_parameters(tool)

    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters=types.Schema(**parameters),
    )


def _translate_parameters(tool: Any) -> dict[str, Any]:
    schema = _extract_input_schema(tool)
    translated = _translate_with_core(tool, schema)

    if isinstance(translated, Mapping):
        translated = dict(translated)
        if "parameters" in translated:
            schema = translated["parameters"]
        elif "functionDeclarations" in translated:
            declarations = translated.get("functionDeclarations") or []
            if declarations:
                schema = declarations[0].get("parameters", schema)
        else:
            schema = translated

    # Gemini accepts a strict JSON Schema subset. Flatten local nested $ref first so
    # removing definitions/$defs later does not leave dangling references behind.
    return _sanitize_gemini_schema(schema)


def _translate_with_core(tool: Any, schema: dict[str, Any]) -> Any:
    if SchemaTranslator is None:
        return None

    translators = []
    to_gemini = getattr(SchemaTranslator, "to_gemini", None)
    if to_gemini is not None:
        translators.append(to_gemini)

    try:
        translator_instance = SchemaTranslator()
    except TypeError:
        translator_instance = None
    if translator_instance is not None:
        instance_to_gemini = getattr(translator_instance, "to_gemini", None)
        if instance_to_gemini is not None:
            translators.append(instance_to_gemini)

    for translator in translators:
        for candidate in (tool, schema):
            try:
                return translator(candidate)
            except TypeError:
                continue
    return None


def _extract_input_schema(tool: Any) -> dict[str, Any]:
    schema = (
        _get_value(tool, "inputSchema")
        or _get_value(tool, "input_schema")
        or _get_value(tool, "parameters")
    )

    if not isinstance(schema, Mapping):
        return {"type": "object", "properties": {}}

    return dict(schema)


def _sanitize_gemini_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        return {"type": "object", "properties": {}}

    copied = deepcopy(dict(schema))
    dereferenced = _resolve_local_refs(copied, copied)
    stripped = _strip_unsupported_schema_keys(dereferenced)

    if not isinstance(stripped, Mapping):
        return {"type": "object", "properties": {}}

    cleaned = dict(stripped)
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned


def _resolve_local_refs(schema: Any, root: Mapping[str, Any], seen: set[str] | None = None) -> Any:
    seen = seen or set()

    if isinstance(schema, list):
        return [_resolve_local_refs(item, root, seen.copy()) for item in schema]

    if not isinstance(schema, Mapping):
        return schema

    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _lookup_local_ref(root, ref)
        if target is not None and ref not in seen:
            merged = deepcopy(target)
            overrides = {key: value for key, value in schema.items() if key != "$ref"}
            if isinstance(merged, Mapping):
                merged = {**merged, **overrides}
            seen.add(ref)
            return _resolve_local_refs(merged, root, seen)

    return {
        key: _resolve_local_refs(value, root, seen.copy())
        for key, value in schema.items()
        if key != "$ref"
    }


def _lookup_local_ref(root: Mapping[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        return None

    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _strip_unsupported_schema_keys(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_unsupported_schema_keys(item) for item in value]

    if isinstance(value, Mapping):
        return {
            key: _strip_unsupported_schema_keys(child)
            for key, child in value.items()
            if key not in UNSUPPORTED_GEMINI_SCHEMA_KEYS
        }

    return value


async def _render_result_for_gemini(result: Any) -> str:
    rendered = await _render_with_core(result)
    if rendered is not None:
        return rendered
    return _render_content_to_string(result)


async def _render_with_core(result: Any) -> str | None:
    if ResultRenderer is None:
        return None

    renderers = []
    to_gemini_part = getattr(ResultRenderer, "to_gemini_part", None)
    if to_gemini_part is not None:
        renderers.append(to_gemini_part)

    try:
        renderer_instance = ResultRenderer()
    except TypeError:
        renderer_instance = None
    if renderer_instance is not None:
        instance_to_gemini_part = getattr(renderer_instance, "to_gemini_part", None)
        if instance_to_gemini_part is not None:
            renderers.append(instance_to_gemini_part)

    for renderer in renderers:
        try:
            rendered = await _maybe_await(renderer(result))
        except TypeError:
            continue
        return _extract_rendered_text(rendered)
    return None


def _extract_rendered_text(rendered: Any) -> str | None:
    if rendered is None:
        return None
    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, Mapping):
        if "result" in rendered:
            return _stringify(rendered["result"])
        response = rendered.get("response")
        if isinstance(response, Mapping) and "result" in response:
            return _stringify(response["result"])
        function_response = rendered.get("function_response") or rendered.get("functionResponse")
        if isinstance(function_response, Mapping):
            return _extract_rendered_text(function_response)
    return _stringify(rendered)


def _render_content_to_string(result: Any) -> str:
    if isinstance(result, str):
        return result

    content = _get_value(result, "content")
    if content is None and isinstance(result, Mapping):
        content = result.get("result") or result.get("structuredContent")

    if isinstance(content, list):
        rendered_items = [_render_content_item(item) for item in content]
        return "\n".join(item for item in rendered_items if item)

    if content is not None:
        return _stringify(content)

    return _stringify(result)


def _render_content_item(item: Any) -> str:
    if isinstance(item, str):
        return item

    text = _get_value(item, "text")
    if text is not None:
        return str(text)

    data = _get_value(item, "data")
    if data is not None:
        return _stringify(data)

    return _stringify(item)


def _coerce_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, Mapping):
        return dict(args)
    if hasattr(args, "to_dict"):
        return dict(args.to_dict())
    if hasattr(args, "items"):
        return dict(args.items())
    if hasattr(args, "fields"):
        return {key: _proto_value_to_python(value) for key, value in args.fields.items()}
    if hasattr(args, "model_dump"):
        return dict(args.model_dump())
    return dict(args)


def _proto_value_to_python(value: Any) -> Any:
    which = value.WhichOneof("kind") if hasattr(value, "WhichOneof") else None
    if which == "null_value":
        return None
    if which == "number_value":
        return value.number_value
    if which == "string_value":
        return value.string_value
    if which == "bool_value":
        return value.bool_value
    if which == "struct_value":
        return {key: _proto_value_to_python(child) for key, child in value.struct_value.fields.items()}
    if which == "list_value":
        return [_proto_value_to_python(child) for child in value.list_value.values]
    return value


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
