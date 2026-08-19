"""
Tool handler node (Story 03).

Executes a requested classroom action through the validated tool registry.

The node:
1. Reads the requested tool name and arguments from state.
2. Validates the arguments using the corresponding Pydantic schema.
3. Executes the tool from TOOL_REGISTRY.
4. Logs the tool call in state["tool_calls"].
5. Stores the result as the final answer.

Tools are never executed directly from free-form LLM output without
schema validation.
"""

from datetime import datetime, timezone

from pydantic import ValidationError

from graph.state import TutorState
from tools.schemas import TOOL_SCHEMAS
from tools.registry import TOOL_REGISTRY


def tool_handler_node(state: TutorState) -> TutorState:
    """
    LangGraph node for safely executing a classroom tool.

    Expected state fields:
        requested_tool: str
        tool_arguments: dict
    """

    tool_name = state.get("requested_tool")
    tool_arguments = state.get("tool_arguments", {})

    # ---------------------------------------------------------
    # 1. Check that a tool was actually requested
    # ---------------------------------------------------------
    if not tool_name:
        state["final_answer"] = "No tool action was requested."
        return state

    # ---------------------------------------------------------
    # 2. Check that the requested tool exists
    # ---------------------------------------------------------
    if tool_name not in TOOL_REGISTRY:
        result = {
            "status": "error",
            "message": f"Unknown tool: {tool_name}",
        }

        state["tool_calls"].append(
            {
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        state["final_answer"] = result["message"]
        return state

    # ---------------------------------------------------------
    # 3. Get the Pydantic schema for validation
    # ---------------------------------------------------------
    schema_class = TOOL_SCHEMAS.get(tool_name)

    if schema_class is None:
        result = {
            "status": "error",
            "message": f"No validation schema found for tool: {tool_name}",
        }

        state["tool_calls"].append(
            {
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        state["final_answer"] = result["message"]
        return state

    # ---------------------------------------------------------
    # 4. Validate arguments
    # ---------------------------------------------------------
    try:
        validated_args = schema_class(**tool_arguments)

    except ValidationError as error:
        result = {
            "status": "validation_error",
            "message": "The tool arguments were invalid.",
            "details": error.errors(),
        }

        state["tool_calls"].append(
            {
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        state["final_answer"] = (
            f"Tool '{tool_name}' could not run because its arguments "
            "were invalid."
        )

        return state

    # ---------------------------------------------------------
    # 5. Execute the validated tool
    # ---------------------------------------------------------
    try:
        tool_function = TOOL_REGISTRY[tool_name]

        result = tool_function(validated_args)

    except Exception as error:
        result = {
            "status": "error",
            "message": f"Tool execution failed: {str(error)}",
        }

    # ---------------------------------------------------------
    # 6. Log every tool call
    # ---------------------------------------------------------
    state["tool_calls"].append(
        {
            "tool_name": tool_name,
            "arguments": tool_arguments,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    # ---------------------------------------------------------
    # 7. Return a student-facing result
    # ---------------------------------------------------------
    if isinstance(result, dict):
        state["final_answer"] = (
            result.get("message")
            or result.get("status")
            or str(result)
        )
    else:
        state["final_answer"] = str(result)

    return state