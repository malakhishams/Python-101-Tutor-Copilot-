"""
Tool schemas for Story 03 — Safe Actions via Function Calling.

The tutor is only allowed to request actions represented by these
explicit schemas. Tool arguments are validated before execution.
"""

from typing import Literal
from pydantic import BaseModel, Field


class CreatePracticeQuizArgs(BaseModel):
    topic: str = Field(
        ...,
        description="Python topic for the practice quiz, e.g. lists or loops.",
    )
    num_questions: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of quiz questions to create.",
    )
    difficulty: Literal["beginner"] = Field(
        default="beginner",
        description="Quiz difficulty. This tutor only supports beginner level.",
    )


class RecommendExercisesArgs(BaseModel):
    topic: str = Field(
        ...,
        description="Python topic the student wants to practice.",
    )
    num_exercises: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of exercises to recommend.",
    )


class LogStudentQuestionArgs(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The student's question to log.",
    )
    topic: str = Field(
        ...,
        min_length=1,
        description="Topic associated with the question.",
    )


class EscalateToTAArgs(BaseModel):
    reason: str = Field(
        ...,
        min_length=1,
        description="Reason the question should be escalated.",
    )
    question: str = Field(
        ...,
        min_length=1,
        description="The student's question.",
    )


# Registry of every action the tutor is allowed to execute.
TOOL_SCHEMAS = {
    "create_practice_quiz": CreatePracticeQuizArgs,
    "recommend_exercises": RecommendExercisesArgs,
    "log_student_question": LogStudentQuestionArgs,
    "escalate_to_ta": EscalateToTAArgs,
}