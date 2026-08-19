"""
Controlled classroom action: log_student_question.

Story 03:
Records a student's question in a structured form.

For the demo, this function does NOT persist student PII or write to
an external database. It simply returns a structured event that can
be captured in the application trace.
"""

from tools.schemas import LogStudentQuestionArgs


def log_student_question(args: LogStudentQuestionArgs) -> dict:
    """Log a student question as a structured event."""

    return {
        "status": "logged",
        "action": "log_student_question",
        "topic": args.topic,
        "question": args.question,
    }