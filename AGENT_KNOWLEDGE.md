## Integration test patterns (feature-1280786-0368)

**Files**: tests/conftest.py, tests/test_integration.py

**Key findings**:
- `respx.mock` patches `httpx.AsyncClient` at the class level — it must be active WHEN the `BackendManager._client` is created. Tests that create a `BackendManager` before entering a respx context will have an unpatched client. Solution: use `httpx.AsyncClient.post` mock via `unittest.mock.AsyncMock` directly on the manager instance, OR use a module-level `respx.mock(assert_all_called=False)` created before any `BackendManager`.
- `make_chat_response` must raise `httpx.HTTPStatusError` for non-2xx responses so `app.py`'s `except httpx.HTTPStatusError as e:` handler fires → 502. Without the check, `.json()` on an empty 503 body throws `JSONDecodeError`.
- `httpx.Response(json={...})` creates a proper response — `MagicMock` breaks when `make_chat_response` accesses `.status_code`.
- Use `.venv/bin/python` (not system `python3`) so `respx` is in the path.
- Concurrency test: 3 requests with 1s delay each must complete in ≤5s to prove queueing, not dropping. Check `start_times[i] >= end_times[i-1]` for strict sequential ordering.
