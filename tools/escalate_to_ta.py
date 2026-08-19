"""
Controlled classroom action: escalate_to_ta.

Story 03:
Provides a safe mechanism for escalating a question that the tutor
cannot confidently answer.

For the demo, no real TA ticket is created. A structured escalation
event is returned instead.
"""

from tools.schemas import EscalateToTAArgs


def escalate_to_ta(args: EscalateToTAArgs) -> dict:
    """Create a structured TA escalation request."""

    return {
        "status": "escalated",
        "action": "escalate_to_ta",
        "reason": args.reason,
        "question": args.question,
        "message": "The question has been marked for TA review.",
    }