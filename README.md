# Python 101 Tutor Copilot

An agentic RAG-based tutoring assistant designed to help beginner Python students learn from course material.

The system receives a student's question, plans how to answer it, retrieves relevant information from a Python course textbook, generates a beginner-friendly response, reflects on the answer for quality and grounding, and either finalizes the answer or retries the workflow.

It also supports structured tools for practice quizzes, exercise recommendations, question logging, and TA escalation.

Built for the Sprints Advanced Agentic AI course (Task 3).

---

## Features

- Agentic workflow built with **LangGraph**
- RAG pipeline using **Qdrant**
- PDF ingestion and chunking
- Semantic search using sentence embeddings
- Beginner-friendly Level 1 Python explanations
- Citation metadata for retrieved textbook content
- Reflection and bounded retry loop
- Clarification path for incomplete debugging questions
- Structured tool calling
- Practice quiz creation
- Exercise recommendations
- Student question logging
- TA escalation
- **Langfuse** tracing and observability

---

# Architecture

The main workflow is:

```text
                    ┌──────────────┐
                    │ Intake / Plan│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ ReAct Router │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
     ┌──────────────────┐      ┌─────────────────────┐
     │ Clarification    │      │ Retrieve from Qdrant│
     │ Question         │      └──────────┬──────────┘
     └────────┬─────────┘                 │
              │                           ▼
              │                    ┌──────────────┐
              │                    │ Draft Answer │
              │                    └──────┬───────┘
              │                           │
              │                           ▼
              │                    ┌──────────────┐
              │                    │  Reflection  │
              │                    └──────┬───────┘
              │                           │
              │              ┌────────────┴────────────┐
              │              │                         │
              │              ▼                         ▼
              │         Retry Retrieval             Final Answer
              │
              ▼
             END
```

---

# Project Structure

```text
Python-101-Tutor-Copilot/
│
├── data/
│   └── course_pack.pdf
│
├── demo/
│   ├── demo.md
│   └── screenshots/
│       ├── 01_end_to_end_rag.png
│       ├── 02_qdrant_ingestion.png
│       ├── 03_qdrant_health_check.png
│       ├── 04_tool_calling_quiz.png
│       ├── 05_clarification_path.png
│       └── 06_langfuse_trace.png
│
├── graph/
│   ├── nodes/
│   │   ├── intake_plan.py
│   │   ├── react_router.py
│   │   ├── retrieve.py
│   │   ├── draft.py
│   │   ├── reflect.py
│   │   ├── route_decision.py
│   │   └── tool_handler.py
│   │
│   ├── build_graph.py
│   └── state.py
│
├── ingestion/
│   ├── chunker.py
│   └── embed_and_index.py
│
├── tools/
│   ├── schemas.py
│   └── registry.py
│
├── run.py
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── .gitignore
└── README.md
```

> The exact file structure may vary slightly depending on the local implementation.

---

# How It Works

## 1. Intake and Planning

The system first analyzes the student's question and determines its intent.

Supported intents include:

- `concept`
- `debug`
- `quiz_prep`

It then creates a simple plan describing how it will approach the question.


---

## 2. ReAct-Style Routing

The router determines whether the system has enough information to continue.

For example, if a student asks for debugging help but does not provide their code or error details, the system does not attempt to guess.

Instead, it routes the request to a clarification path and asks for the missing information.


---

## 3. Retrieval-Augmented Generation

The student's question is converted into an embedding and used to search the Qdrant vector database.

The project uses the collection:

```text
python101_textbook
```

The course PDF is processed into chunks and embedded before being stored in Qdrant.

The ingestion process successfully indexed:

```text
787 chunks
```

Retrieved chunks include metadata such as:

- Text
- Page number
- Chapter
- Chunk ID
- Similarity score

---

## 4. Answer Generation

The retrieved textbook content is passed to the drafting node.

The system generates an answer that is designed to be:

- Grounded in the retrieved textbook excerpts
- Appropriate for Level 1 Python students
- Easy to understand
- Supported by a small code example

Example question:

```text
What is a Python list?
```

The system retrieves relevant textbook sections and generates a beginner-friendly explanation with citation information.

---

## 5. Reflection and Retry

Before an answer is returned, it passes through a reflection node.

The reflection checks four areas:

### Correctness

Every factual claim should be supported by the retrieved textbook excerpts.

### Tone

The answer should use beginner-friendly language and avoid unnecessary jargon.

### Example Match

The code example should demonstrate the concept being explained.

### Grounding

The response should rely on the retrieved textbook material and avoid unsupported claims.

The reflection returns one of two verdicts:

```text
VERDICT: pass
```

or:

```text
VERDICT: fail
FEEDBACK: <specific instructions for fixing the answer>
```

If the answer fails reflection, the graph can retry the retrieval and drafting process.

The retry loop is bounded to prevent infinite execution.

---

# Tool Calling

The Tutor Copilot supports structured tools for student assistance.

## Create Practice Quiz

Creates a beginner-friendly practice quiz for a selected Python topic.

Example request:

```text
Create a 3-question beginner quiz about Python lists
```

Example result:

```python
{
    "status": "created",
    "action": "create_practice_quiz",
    "topic": "lists",
    "num_questions": 3,
    "difficulty": "beginner"
}
```

---

## Recommend Exercises

Recommends practice exercises for a Python topic.

Example result:

```python
{
    "status": "success",
    "action": "recommend_exercises",
    "topic": "loops",
    "num_exercises": 3
}
```

---

## Log Student Question

Records information about a student's question.

Example result:

```python
{
    "status": "logged",
    "action": "log_student_question",
    "topic": "loops",
    "question": "Why does my loop not stop?"
}
```

---

## Escalate to TA

Marks a question for TA review when additional assistance is needed.

Example result:

```python
{
    "status": "escalated",
    "action": "escalate_to_ta",
    "reason": "Student needs help with debugging",
    "question": "Why does my loop not stop?"
}
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Python-101-Tutor-Copilot
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running Qdrant

The application requires a running Qdrant instance.

The project connects to:

```text
http://localhost:6333
```

Make sure Qdrant is running before starting the application or performing ingestion.

---

# Indexing the Course Material

Place the course PDF inside:

```text
data/course_pack.pdf
```

Then run the ingestion pipeline from the project root:

```bash
python -m ingestion.embed_and_index
```

The ingestion process will:

1. Load the PDF
2. Extract pages
3. Split the content into chunks
4. Generate embeddings
5. Create the Qdrant collection
6. Upload the vectors

After successful ingestion, the output should indicate that the collection was created and the chunks were indexed.


---

# Running the Tutor

Ask a concept question:

```bash
python run.py "What is a Python list?"
```

Ask for a practice quiz:

```bash
python run.py "Create a 3-question beginner quiz about Python lists"
```

Example output:

```text
=== PLAN ===
- Retrieve the relevant textbook section for reference
- Summarize the key points at Level 1
- Offer a tiny practice example
- Reflect on accuracy and tone, refine if needed

=== FINAL ANSWER ===
...
```

---

# Observability

The project uses **Langfuse** to trace the execution of the tutoring workflow.

Traces provide visibility into important stages such as:

- Intake and planning
- Retrieval
- Reflection
- Input and output data
- Execution latency

This makes it easier to inspect and debug the agent workflow.

---

# Demo

Detailed screenshots demonstrating the project are available in:

```text
demo/demo.md
```

The demo includes:

1. End-to-end RAG workflow
2. Qdrant ingestion
3. Qdrant health check
4. Tool calling
5. Clarification routing
6. Langfuse tracing

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application language |
| LangGraph | Agent workflow orchestration |
| Qdrant | Vector database |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| Langfuse | Tracing and observability |
| Pydantic | Tool argument validation |

---

# Example Workflow

```text
Student Question
       │
       ▼
Intent + Plan
       │
       ▼
ReAct Router
       │
       ├── Missing Context ──► Ask Clarifying Question
       │
       ▼
Retrieve Textbook Chunks
       │
       ▼
Generate Draft Answer
       │
       ▼
Reflection
       │
       ├── Fail ──► Retry (bounded)
       │
       ▼
Final Answer
```

---

# Limitations

Current limitations include:

- The quality of answers depends on the content available in the indexed course material.
- Retrieval depends on the similarity between the student's question and the stored textbook chunks.
- Some vague questions may still require additional clarification.
- The current tool implementations can be extended with more advanced functionality.
- The application requires Qdrant to be running before retrieval can work.

---

# Future Improvements

Possible future improvements include:

- A web-based chat interface
- Conversation memory
- More advanced tool selection
- Generated quiz questions and answers
- Persistent student progress tracking
- More detailed retrieval evaluation
- Automated tests for graph nodes
- Streaming responses
- Better clarification handling for ambiguous concept questions
- Deployment as a web service

---