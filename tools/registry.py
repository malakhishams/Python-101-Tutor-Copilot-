"""
Central registry of all allowed classroom actions.

The LLM/application can only execute functions present in this registry.
"""

from tools.create_practice_quiz import create_practice_quiz
from tools.recommend_exercises import recommend_exercises
from tools.log_student_question import log_student_question
from tools.escalate_to_ta import escalate_to_ta


TOOL_REGISTRY = {
    "create_practice_quiz": create_practice_quiz,
    "recommend_exercises": recommend_exercises,
    "log_student_question": log_student_question,
    "escalate_to_ta": escalate_to_ta,
}