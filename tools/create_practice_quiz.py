"""
Controlled classroom action: create_practice_quiz.

Story 03:
The tutor can request a practice quiz through this explicitly defined
action. The arguments are validated by the Pydantic schema before this
function is called.
"""

from tools.schemas import CreatePracticeQuizArgs


def create_practice_quiz(args: CreatePracticeQuizArgs) -> dict:
    """Create a beginner practice quiz request.

    This demo implementation does not persist anything to an LMS.
    It returns a structured result that can be logged and displayed.
    """

    return {
        "status": "created",
        "action": "create_practice_quiz",
        "topic": args.topic,
        "num_questions": args.num_questions,
        "difficulty": args.difficulty,
        "message": (
            f"Created a {args.num_questions}-question "
            f"beginner practice quiz about {args.topic}."
        ),
    }