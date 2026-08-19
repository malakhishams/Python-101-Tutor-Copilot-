"""
Langfuse setup for the Python 101 Tutor Copilot.

Provides a small wrapper around the Langfuse Python SDK so the rest of
the application can create traces and observations without depending on
Langfuse configuration details.

If Langfuse credentials are missing, tracing is disabled and the Tutor
Copilot continues to run normally.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def langfuse_is_configured() -> bool:
    """Return True only when the required Langfuse credentials exist."""
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


def get_langfuse_client():
    """
    Return the shared Langfuse client.

    Returns None when Langfuse is not configured so observability remains
    optional during local development.
    """
    if not langfuse_is_configured():
        return None

    from langfuse import get_client

    return get_client()


def check_langfuse_connection() -> bool:
    """
    Check whether the Langfuse client can authenticate.

    Returns False instead of crashing the Tutor Copilot if the credentials
    or connection are invalid.
    """
    client = get_langfuse_client()

    if client is None:
        return False

    try:
        return bool(client.auth_check())
    except Exception as exc:
        print(f"Langfuse connection check failed: {exc}")
        return False


def trace_event(
    name: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: dict[str, Any] | None = None,
    observation_type: str = "span",
) -> None:
    """
    Create one standalone Langfuse observation.

    This is a lightweight helper for logging important workflow events
    such as planning, retrieval, tool calls, reflection, and routing.

    If Langfuse is not configured or temporarily unavailable, the Tutor
    Copilot continues without failing.
    """
    client = get_langfuse_client()

    if client is None:
        return

    try:
        with client.start_as_current_observation(
            as_type=observation_type,
            name=name,
            input=input_data,
        ) as observation:

            update_data = {}

            if output_data is not None:
                update_data["output"] = output_data

            if metadata is not None:
                update_data["metadata"] = metadata

            if update_data:
                observation.update(**update_data)

    except Exception as exc:
        # Observability should never break the main tutor workflow.
        print(f"Langfuse tracing warning for '{name}': {exc}")


def score_current_trace(
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """
    Attach a numeric evaluation score (0-1) to the currently active
    Langfuse trace. Used by observability/eval_set.py to record
    groundedness / clarity / correctness per Story 07.

    No-ops if Langfuse isn't configured, same fail-open pattern as the
    rest of this module -- evaluation should never crash a run.
    """
    client = get_langfuse_client()

    if client is None:
        return

    try:
        client.score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:
        print(f"Langfuse scoring warning for '{name}': {exc}")


def flush_langfuse() -> None:
    """
    Flush pending Langfuse events.

    Important for this project because run.py is a short-lived CLI process.
    """
    client = get_langfuse_client()

    if client is None:
        return

    try:
        client.flush()
    except Exception as exc:
        print(f"Langfuse flush warning: {exc}")