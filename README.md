# Flask2 – LLM Core & Prompt Manager

This repository is part of the **AutiMate AI** project — a multimodal assistive system designed to support neurodivergent individuals. `flask2` specifically hosts the backend Flask server for the **LLM core**, which processes user queries enriched with affective and contextual signals.

---

## Project Purpose

The LLM Core handles:
- Affective prompt conditioning using emotion tags (from FER/SER)
- User profiling for personalized responses
- Response generation using a fine-tuned GPT-Neo model
- RESTful API endpoints for real-time interaction with frontend clients (e.g., Flutter app)
- Note: This is part of the AutiMate smart glasses project.
