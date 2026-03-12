# General-Purpose Analytical Chatbot

A natural-language data analysis chatbot built with Streamlit, AWS Bedrock (Claude),
and pandas. Upload any CSV file and ask questions in plain English. The chatbot
generates tables, charts, and explanations automatically.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [SageMaker and Bedrock Integration](#sagemaker-and-bedrock-integration)
- [Data Flow](#data-flow)
- [Architecture and Flow Diagrams](#architecture-and-flow-diagrams)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features](#features)
- [Supported Operations](#supported-operations)
- [Chart Types](#chart-types)
- [Few-Shot and Meta-Prompt Training](#few-shot-and-meta-prompt-training)
- [Knowledge Base](#knowledge-base)
- [Custom Instructions](#custom-instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

This chatbot provides a conversational interface for exploring and analysing
tabular data. It uses a three-stage pipeline:

1. **Planner** -- Translates the user's natural-language question into a
   structured JSON execution plan using Claude via AWS Bedrock.
2. **Executor** -- Runs the JSON plan against the in-memory pandas DataFrame,
   producing tables, aggregations, or Plotly charts.
3. **Explainer** -- Summarises the result into a short natural-language
   narrative using Claude.

The application runs inside AWS SageMaker Studio and communicates with
AWS Bedrock for all language model operations. No local GPU or model
hosting is required.

### Key Capabilities

- **Natural Language Queries** -- Ask questions in plain English
- **19 Operation Types** -- Count, group, filter, pivot, correlate, and more
- **10 Chart Types** -- Bar, line, scatter, pie, histogram, box, and more
- **Follow-Up Questions** -- Conversation context tracks previous exchanges
- **Relative Date Filtering** -- Supports "last 30 days", "past 6 months", "yesterday"
- **Custom Instructions** -- Column aliases, value mappings, business rules
- **Knowledge Base** -- Store domain facts for context-aware analysis
- **Few-Shot Training** -- Teach the model by example for better accuracy
- **General Questions** -- Time, date, greetings handled without API calls

---

## Architecture

The application is split into 15 Python files organised into five layers:
