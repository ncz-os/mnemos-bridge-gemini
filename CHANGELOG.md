# Changelog

## [0.2.0] — 2026-05-05

### Migrated

- Migrated the Gemini adapter and integration test from `google-generativeai` to `google-genai`.
- Returned typed `google.genai.types.Tool` declarations and `types.Part` function responses.

### Breaking

- v0.1.x users must update imports and generation calls from `google.generativeai`/`GenerativeModel` to `google.genai`/`genai.Client`.

## v0.1.0 - 2026-05-03

- Initial scaffold for the Google Gemini adapter.
- Added MCP tool translation into Gemini `functionDeclarations`.
- Added Gemini `functionCall` to MCP `call_tool` dispatch support.
- Added offline translator and function-call tests plus a guarded integration test.
