# Demo

This document demonstrates the main features of the **Python 101 Tutor Copilot**.

The screenshots below show the end-to-end RAG workflow, Qdrant ingestion and health checks, tool calling, clarification handling, and Langfuse observability.

---

## 1. End-to-End RAG Pipeline

The application receives a student's question, creates a plan, retrieves relevant textbook content from Qdrant, generates a beginner-friendly answer, reflects on the answer, and returns the final response with citations.

Example question:

> What is a Python list?

The output includes:

- A step-by-step plan
- Retrieval from the indexed textbook
- A Level 1 beginner-friendly explanation
- A small Python example
- Citation metadata for the retrieved textbook chunks
- Reflection and retry handling

![End-to-End RAG Demo](demo/screenshots/01_end_to_end_rag.png)

---

## 2. PDF Ingestion and Qdrant Indexing

The course PDF is processed and split into chunks before being embedded with the `all-MiniLM-L6-v2` embedding model.

The generated vectors are then stored in the Qdrant collection:

`python101_textbook`

The ingestion process successfully indexed **787 chunks** from the course material.

![Qdrant Ingestion](demo/screenshots/02_qdrant_ingestion.png)

---

## 3. Qdrant Health Check

Before retrieval, the project verifies that the Qdrant vector database is available and that the required collection can be accessed.

This confirms that the vector database is ready to support semantic retrieval for the tutor.

![Qdrant Health Check](demo/screenshots/03_qdrant_health_check.png)

---

## 4. Tool Calling

The Tutor Copilot supports structured tool calls for student assistance.

The implemented tools include:

- `create_practice_quiz`
- `recommend_exercises`
- `log_student_question`
- `escalate_to_ta`

The example below demonstrates a request to create a beginner-level practice quiz about Python lists.

![Tool Calling Quiz](demo/screenshots/04_tool_calling_quiz.png)

---

## 5. Clarification Path

The ReAct-style router checks whether the student's question contains enough information to continue.

For example, if a student asks for debugging help but does not provide their code or error message, the system does not attempt to guess.

Instead, it routes the request to the clarification path and asks the student to provide the missing information.

![Clarification Path](demo/screenshots/05_clarification_path.png)

---

## 6. Langfuse Observability

Langfuse is used to trace the execution of the Tutor Copilot.

The trace provides visibility into important stages of the graph, including:

- `intake_plan`
- `retrieval`
- `reflection`

This makes it easier to inspect the agent workflow, monitor latency, and review the input and output of the application.

![Langfuse Trace](demo/screenshots/06_langfuse_trace.png)

---

## Summary

The Python 101 Tutor Copilot demonstrates an agentic RAG workflow built around:

- **LangGraph** for workflow orchestration
- **Qdrant** for vector storage and semantic retrieval
- **Sentence Transformers** for embeddings
- **Structured tools** for quiz creation, exercise recommendations, logging, and escalation
- **Reflection and retry logic** for answer quality control
- **Clarification routing** when the student has not provided enough context
- **Langfuse** for tracing and observability

Together, these components create a beginner-focused Python tutoring assistant that retrieves information from the course material, generates grounded answers, validates its responses, and handles different student requests through an agentic workflow.