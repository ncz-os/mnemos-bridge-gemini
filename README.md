# mnemos-bridge-gemini

Google Gemini adapter for the `mnemos-bridge-core` MCP bridge abstraction. It translates MCP tool definitions into Gemini `functionDeclarations` and dispatches Gemini `functionCall` parts back to MCP tools.

## Install

```bash
pip install "mnemos-bridge-gemini>=0.2.0" "google-genai>=1.0"
```

## Quick Start

```python
from google import genai
from google.genai import types

from mnemos_bridge_gemini import MnemosGeminiAdapter


async def main() -> None:
    client = genai.Client(api_key="...")

    adapter = await MnemosGeminiAdapter.connect(
        "http://192.168.207.67:5003",
        "...mcp token...",
    )

    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text="Search MNEMOS for memories about infrastructure")],
            )
        ]
        config = types.GenerateContentConfig(tools=await adapter.gemini_tools())

        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config,
        )
        function_call = response.candidates[0].content.parts[0].function_call

        function_response_part = await adapter.handle_function_call(function_call)
        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(role="user", parts=[function_response_part])
        )

        final = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=config,
        )
        print(final.text)
    finally:
        await adapter.aclose()
```

## Multi-Turn Example

```python
from google import genai
from google.genai import types

from mnemos_bridge_gemini import MnemosGeminiAdapter


async def run_loop(prompt: str) -> str:
    client = genai.Client(api_key="...")

    async with await MnemosGeminiAdapter.connect(
        "http://192.168.207.67:5003",
        "...mcp token...",
    ) as adapter:
        model = "gemini-2.0-flash"
        config = types.GenerateContentConfig(tools=await adapter.gemini_tools())
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        for _ in range(8):
            parts = response.candidates[0].content.parts
            function_calls = [
                part.function_call
                for part in parts
                if getattr(part, "function_call", None)
                and getattr(part.function_call, "name", "")
            ]
            if not function_calls:
                return response.text

            response_parts = []
            for function_call in function_calls:
                response_parts.append(await adapter.handle_function_call(function_call))

            contents.append(response.candidates[0].content)
            contents.append(types.Content(role="user", parts=response_parts))
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

        return response.text
```

## Gemini Schema Subset

Gemini function parameters use a strict JSON Schema subset. This adapter flattens local nested `$ref` references before stripping keywords Gemini does not accept:

`additionalProperties`, `oneOf`, `not`, `anyOf`, `allOf`, `if`, `then`, `else`, `patternProperties`, `definitions`, `$defs`, `contentEncoding`, `contentMediaType`, `deprecated`, `readOnly`, and `writeOnly`.

When `mnemos_bridge_core.SchemaTranslator.to_gemini()` is available, the adapter delegates schema translation to it and then applies the same conservative cleanup pass. See the Gemini function calling documentation: https://ai.google.dev/gemini-api/docs/function-calling

## v0.1 to v0.2 Migration

Version 0.2.0 migrates from the deprecated `google-generativeai` package to the
current `google-genai` package.

- Replace `import google.generativeai as genai` with `from google import genai`
  and import `types` from `google.genai`.
- Replace `genai.configure(...)` and `genai.GenerativeModel(...)` with
  `genai.Client(...)` and `client.aio.models.generate_content(...)`.
- Pass tools through `types.GenerateContentConfig(tools=await adapter.gemini_tools())`.
- `adapter.handle_function_call(...)` now returns a `types.Part` containing a
  `types.FunctionResponse`; append it to the next user turn's `parts` list.

## Vertex AI Usage

The same Gemini SDK flow can be used with Vertex AI credentials. Set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON file and configure the SDK for Vertex AI:

```python
from google import genai

client = genai.Client(vertexai=True, project="your-project", location="us-central1")
```

Then use `client.aio.models.generate_content(...)` and pass tools through
`types.GenerateContentConfig(tools=await adapter.gemini_tools())` as in the examples above.

## Testing

Run offline unit tests without real Google or MNEMOS services:

```bash
pytest tests/test_translator_offline.py tests/test_handle_function_call_offline.py
```

Run the guarded integration test by setting all required service environment variables:

```bash
export GOOGLE_API_KEY=...
export MNEMOS_TEST_BASE=http://192.168.207.67:5003
export MNEMOS_MCP_TOKEN=...
export MNEMOS_TEST_GEMINI_MODEL=gemini-2.0-flash
pytest tests/integration/test_gemini_tool_loop.py
```
