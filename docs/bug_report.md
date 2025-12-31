# Bug Report and Fixes

## Critical Issues

### 1. Chainlit Crash on Agent Response
**Severity:** Critical
**Description:** The Chainlit app crashed with `ValueError: not enough values to unpack (expected 2, got 1)` when processing the agent's response.
**Root Cause:**
* `src/chainlit_app.py` expected the agent to return `intermediate_steps` as a list of tuples `(AgentAction, observation)`.
* `agent.py` was returning a dictionary or object incompatible with the unpacking loop.
* `agent.py` was using `create_agent` from `langchain.agents` but passing arguments intended for `langgraph`.

**Fix:**
* **Refactored `src/agent.py`**: Switched to `langgraph.prebuilt.create_react_agent`.
* **Refactored output**: The `arun` method now extracts `tool_calls` and outputs from the LangGraph message history and formats them as a list of dictionaries.
* **Updated `src/chainlit_app.py`**: Modified the message processing loop to handle both the legacy tuple format and the new dictionary format.

### 2. TypeError in Agent Initialization
**Severity:** High
**Description:** `TypeError: create_react_agent() got unexpected keyword arguments: {'state_modifier': ...}`
**Fix:** Updated `src/agent.py` to use `prompt=self.SYSTEM_PROMPT`.

### 3. Retriever Argument and Object Access Error
**Severity:** High
**Description:** 
1. `TypeError: HybridRetriever.retrieve() got an unexpected keyword argument 'country'`.
2. Incorrect access of `PolicyChunk` attributes using `.get()` (treating object as dict).

**Root Cause:**
* `agent.py` was calling `self.retriever.retrieve(query, country=country)` but the signature is `retrieve(query, filters={...})`.
* `agent.py` handled the results as dictionaries (`r.get('content')`) but `retrieve` returns `PolicyChunk` dataclass objects (`r.text`).

**Fix:**
* Updated `search_policies` tool in `src/agent.py` to:
    * Construct a `filters` dictionary passed to `retrieve`.
    * Access `PolicyChunk` attributes directly (e.g., `r.text`, `r.document_name`).

## Verification
* A reproduction script `tests/repro_crash.py` confirmed the original crash.
* `tests/verify_fix.py` verified the fix in `chainlit_app.py`.
* Manual testing confirmed the agent now runs without errors.
