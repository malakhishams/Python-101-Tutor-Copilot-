"""
Controlled classroom action: recommend_exercises.

Story 03:
Provides a safe, structured way for the tutor to recommend practice
activities without allowing arbitrary actions.
"""

from tools.schemas import RecommendExercisesArgs


def recommend_exercises(args: RecommendExercisesArgs) -> dict:
    """Generate a structured exercise recommendation.

    This demo implementation does not connect to an external exercise
    database. It returns a deterministic recommendation request.
    """

    return {
        "status": "success",
        "action": "recommend_exercises",
        "topic": args.topic,
        "num_exercises": args.num_exercises,
        "recommendations": [
            {
                "title": f"{args.topic} practice exercise {i}",
                "level": "beginner",
            }
            for i in range(1, args.num_exercises + 1)
        ],
    }