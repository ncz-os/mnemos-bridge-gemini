import os

import pytest


pytestmark = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and os.getenv("MNEMOS_TEST_BASE")),
    reason="requires GOOGLE_API_KEY and MNEMOS_TEST_BASE",
)


@pytest.mark.asyncio
async def test_gemini_tool_loop_requests_search_memories():
    from google import genai
    from google.genai import types

    from mnemos_bridge_gemini import MnemosGeminiAdapter

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    adapter = await MnemosGeminiAdapter.connect(
        os.environ["MNEMOS_TEST_BASE"],
        os.getenv("MNEMOS_MCP_TOKEN", ""),
    )

    try:
        model_name = os.getenv("MNEMOS_TEST_GEMINI_MODEL", "gemini-2.0-flash")
        config = types.GenerateContentConfig(tools=await adapter.gemini_tools())
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text="Search MNEMOS for memories about infrastructure")],
            )
        ]

        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        parts = response.candidates[0].content.parts
        function_calls = [
            part.function_call
            for part in parts
            if getattr(part, "function_call", None)
            and getattr(part.function_call, "name", "")
        ]

        assert any(function_call.name == "search_memories" for function_call in function_calls)

        function_call = next(
            function_call
            for function_call in function_calls
            if function_call.name == "search_memories"
        )
        function_response_part = await adapter.handle_function_call(function_call)
        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[function_response_part],
            )
        )
        final = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        # Some Gemini models chain another tool call instead of emitting
        # text on the second turn. That's legitimate agent behavior — but
        # makes `final.text` raise. The bridge contract is satisfied as
        # soon as one tool round-trip succeeds; we got the function_call,
        # we dispatched it via the adapter, we sent the function_response
        # back. Anything beyond that is model-driven multi-turn outside
        # the bridge's contract.
        final_parts = final.candidates[0].content.parts
        has_text = any(getattr(p, "text", "") for p in final_parts)
        has_followup_tool = any(
            getattr(p, "function_call", None)
            and getattr(p.function_call, "name", "")
            for p in final_parts
        )
        assert has_text or has_followup_tool, (
            "Gemini second turn returned neither text nor a follow-up tool call"
        )
    finally:
        await adapter.aclose()
