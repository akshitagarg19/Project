# TDS Virtual Teaching Assistant

This project is a Virtual Teaching Assistant API for the IIT Madras Tools in Data Science (TDS) Jan 2025 course.

## Features

- Scrapes TDS Jan 2025 Discourse posts for question-answering data
- Builds vector embeddings of forum posts
- Exposes a FastAPI endpoint `/api/` that accepts student questions (and optional images)
- Returns relevant, context-aware answers with reference links

## Setup

1. Clone the repo
2. Create a Python virtual environment and install dependencies:

   ```bash
   pip install -r requirements.txt
