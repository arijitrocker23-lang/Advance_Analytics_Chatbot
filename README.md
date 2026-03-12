# Advanced Analytical Chatbot

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

```mermaid
graph TD
    subgraph UI_LAYER["USER INTERFACE LAYER"]
        direction LR
        APP["app.py<br/>Entry Point"]
        UI["ui_components.py<br/>Rendering"]
        
        subgraph UI_SECTIONS["UI Sections"]
            direction LR
            SIDEBAR["Sidebar"]
            CONFIG_PANEL["Configuration<br/>Panel"]
            CHAT["Chat<br/>Interface"]
        end

        APP --> UI
        UI --> SIDEBAR
        UI --> CONFIG_PANEL
        UI --> CHAT
    end

    subgraph ORCH_LAYER["ORCHESTRATION LAYER"]
        direction LR
        ANALYZER["analyzer.py<br/>Coordinator"]
        CONVERSATION["conversation.py<br/>Context Manager"]
        ANALYZER --- CONVERSATION
    end

    subgraph PIPELINE_LAYER["ANALYSIS PIPELINE"]
        direction LR
        PLANNER["planner.py<br/>Question to<br/>JSON Plan"]
        EXECUTOR["executor.py<br/>JSON Plan to<br/>pandas Operations"]
        EXPLAINER["explainer.py<br/>Result to<br/>Summary"]
        PLANNER -->|"JSON Plan"| EXECUTOR
        EXECUTOR -->|"Structured Result"| EXPLAINER
    end

    subgraph SUPPORT_LAYER["SUPPORT MODULES"]
        direction LR
        BEDROCK["bedrock_client.py<br/>AWS Bedrock<br/>API Wrapper"]
        CHART["chart_engine.py<br/>Plotly Chart<br/>Generation"]
        UTILS["utils.py<br/>JSON / Schema<br/>Dates / Filters"]
        DATA_LOADER["data_loader.py<br/>CSV Loading<br/>S3 Integration"]
    end

    subgraph KNOWLEDGE_LAYER["KNOWLEDGE LAYER"]
        direction LR
        FEW_SHOT["few_shot.py<br/>8 Built-in Examples<br/>8 Meta-Prompts<br/>Custom Training"]
        KB["knowledge_base.py<br/>Domain Facts<br/>Keyword Search<br/>Prompt Injection"]
    end

    subgraph DATA_LAYER["DATA STRUCTURES"]
        direction LR
        DATA_MODELS["data_models.py<br/>CustomInstructions<br/>FewShotExample<br/>KnowledgeBaseEntry"]
        CONFIG["config.py<br/>Model IDs<br/>System Prompts<br/>Settings"]
    end

    %% Layer connections (top to bottom)
    UI_LAYER --> ORCH_LAYER
    ORCH_LAYER --> PIPELINE_LAYER
    PIPELINE_LAYER --> SUPPORT_LAYER
    PIPELINE_LAYER --> KNOWLEDGE_LAYER
    SUPPORT_LAYER --> DATA_LAYER
    KNOWLEDGE_LAYER --> DATA_LAYER

    %% Styling
    classDef uiStyle fill:#4A90D9,stroke:#2C5F8A,color:#FFFFFF,stroke-width:2px
    classDef orchStyle fill:#E67E22,stroke:#A85C15,color:#FFFFFF,stroke-width:2px
    classDef pipeStyle fill:#27AE60,stroke:#1B7A43,color:#FFFFFF,stroke-width:2px
    classDef supportStyle fill:#8E44AD,stroke:#6C3483,color:#FFFFFF,stroke-width:2px
    classDef knowledgeStyle fill:#1ABC9C,stroke:#148F77,color:#FFFFFF,stroke-width:2px
    classDef dataStyle fill:#E74C3C,stroke:#A93226,color:#FFFFFF,stroke-width:2px

    class APP,UI,SIDEBAR,CONFIG_PANEL,CHAT uiStyle
    class ANALYZER,CONVERSATION orchStyle
    class PLANNER,EXECUTOR,EXPLAINER pipeStyle
    class BEDROCK,CHART,UTILS,DATA_LOADER supportStyle
    class FEW_SHOT,KB knowledgeStyle
    class DATA_MODELS,CONFIG dataStyle
````

### Layer Responsibilities

- **User Interface Layer** -- Streamlit widgets, layout, CSS styling, chat
  rendering, and user input handling. No business logic lives here.

- **Orchestration Layer** -- The `Analyzer` class owns the DataFrame, schema,
  and all pipeline components. It wires together planning, execution, and
  explanation into a single `process_question()` call. The `ConversationContext`
  maintains a rolling window of recent exchanges for follow-up resolution.

- **Analysis Pipeline** -- Three sequential stages that transform a question
  into a result:
  - Planner: question text to JSON plan (via Bedrock API)
  - Executor: JSON plan to pandas operations (local computation)
  - Explainer: structured result to natural-language summary (via Bedrock API)

- **Support Modules** -- Shared utilities including the Bedrock API wrapper,
  Plotly chart generation, CSV loading, date parsing, filter application,
  and JSON serialisation helpers.

- **Knowledge Layer** -- In-memory stores for few-shot examples, meta-prompt
  instructions, and domain knowledge entries that are injected into the
  planner prompt for improved accuracy.

- **Data Structures** -- Python dataclasses (`CustomInstructions`,
  `FewShotExample`, `KnowledgeBaseEntry`) and centralised configuration
  (`Config`).

---

## SageMaker and Bedrock Integration

### How It Works

The chatbot runs as a Streamlit application inside an AWS SageMaker Studio
notebook instance. It communicates with AWS Bedrock to access Claude
foundation models for natural-language understanding and generation.

````mermaid
graph LR
    subgraph USER["Web Browser"]
        BROWSER["User Interface<br/>Streamlit UI<br/>Chat + Tables + Charts"]
    end

    subgraph SAGEMAKER["AWS SageMaker Studio"]
        STREAMLIT["Streamlit App<br/>Python Runtime<br/>pandas + plotly"]
    end

    subgraph BEDROCK["AWS Bedrock"]
        CLAUDE["Claude 4.5<br/>Sonnet / Opus<br/>Converse API"]
    end

    subgraph S3["Amazon S3"]
        S3_BUCKET["Optional CSV<br/>Storage"]
    end

    %% Browser to SageMaker
    BROWSER -->|"HTTP Request<br/>Port 8501"| STREAMLIT
    STREAMLIT -->|"HTML / JS<br/>Response"| BROWSER

    %% SageMaker to Bedrock
    STREAMLIT -->|"boto3 converse()<br/>JSON Request"| CLAUDE
    CLAUDE -->|"JSON Response<br/>Plan + Explanation"| STREAMLIT

    %% SageMaker to S3
    STREAMLIT -->|"boto3 get_object()<br/>CSV Download"| S3_BUCKET

    %% Styling
    classDef userStyle fill:#3498DB,stroke:#2471A3,color:#FFFFFF,stroke-width:2px
    classDef sagemakerStyle fill:#FF9900,stroke:#CC7A00,color:#FFFFFF,stroke-width:2px
    classDef bedrockStyle fill:#8C4FFF,stroke:#6B3ACC,color:#FFFFFF,stroke-width:2px
    classDef s3Style fill:#3F8624,stroke:#2D6119,color:#FFFFFF,stroke-width:2px

    class BROWSER userStyle
    class STREAMLIT sagemakerStyle
    class CLAUDE bedrockStyle
    class S3_BUCKET s3Style
````

### Authentication Flow

1. The SageMaker notebook instance has an **IAM execution role** attached.
2. When the application creates a `boto3.client("bedrock-runtime")`, the
   SDK automatically retrieves temporary credentials from the instance
   metadata service via AWS STS.
3. No API keys, access keys, or manual credential configuration is needed.
4. The IAM role must have the `bedrock:InvokeModel` permission for the
   Claude models being used.

### IAM Policy Requirements

The SageMaker execution role needs this policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-*",
                "arn:aws:bedrock:us-east-1:*:inference-profile/us.anthropic.claude-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:HeadObject"
            ],
            "Resource": "arn:aws:s3:::your-bucket-name/*"
        }
    ]
}
````

Bedrock API Usage
The application makes two types of Bedrock API calls per data question:

1. Planner Call -- Converts the question into a JSON plan. Uses temperature 0.0 and max tokens 2048 for deterministic, structured output.
2. Explainer Call -- Converts the structured result into a natural-language summary. Uses temperature 0.2 and max tokens 512 for slightly creative but concise output.
3. General questions (time, date, greetings) are answered locally without
any API calls.

Model Constraint
Claude 4.5 models do not allow both temperature and top_p to be
specified in the same request. The BedrockClient sends only
temperature to avoid ValidationException errors.


## Data Flow
Complete Question Processing Pipeline

````mermaid
flowchart TD
    START(["User Types Question<br/>in Chat Input"])

    subgraph STEP_1["STEP 1: INPUT PROCESSING"]
        direction TB
        Q_INPUT["Question Text"]
        ADD_HISTORY["Add to Chat History<br/>(session_state)"]
        CALL_ANALYZER["analyzer.process_question()"]
    end

    subgraph STEP_2["STEP 2: PLANNING"]
        direction TB
        CHECK_GENERAL{"Is it a General<br/>Question?<br/>(time, date, hello)"}
        GENERAL_ANSWER["Return Canned Answer<br/>(no API call needed)"]
        CHECK_DATA{"Is Dataset<br/>Loaded?"}
        NO_DATA["Return Error:<br/>'No dataset loaded'"]

        subgraph PROMPT_ASSEMBLY["Prompt Assembly"]
            direction TB
            SCHEMA["DataFrame Schema<br/>(columns, types, examples)"]
            DATE_CTX["Current Date/Time<br/>Context"]
            FEW_SHOT_EX["Few-Shot Examples<br/>(8 built-in + custom)"]
            META_PROMPTS["Meta-Prompt<br/>Instructions<br/>(8 built-in + custom)"]
            KB_SEARCH["Knowledge Base<br/>Keyword Search<br/>(top 3 matches)"]
            CUSTOM_INSTR["Custom Instructions<br/>(aliases, rules,<br/>mappings, terminology)"]
            CONV_CTX["Conversation History<br/>(last 5 exchanges)"]
            USER_Q["User Question"]
        end

        BEDROCK_PLAN["AWS Bedrock API Call<br/>Claude Sonnet/Opus 4.5<br/>System: PLANNER_SYSTEM_PROMPT<br/>Temperature: 0.0"]
        PARSE_JSON["Parse JSON Response<br/>Strip markdown fences<br/>json.loads()"]
        JSON_PLAN["Structured JSON Plan<br/>{operation, filters,<br/>groupby, columns, ...}"]
    end

    subgraph STEP_3["STEP 3: EXECUTION"]
        direction TB
        RESOLVE_ALIASES["Resolve Column Aliases<br/>(user terms to actual names)"]
        APPLY_FILTERS["Apply Filters<br/>(eq, contains, isin,<br/>relative_date, between, ...)"]
        RESOLVE_VALUES["Resolve Value Mappings<br/>(user terms to actual values)"]
        FILTERED_DF["Filtered DataFrame"]
        DATE_RANGE["Date Range Metadata<br/>(if relative_date used)"]

        DISPATCH{"Operation<br/>Type?"}

        subgraph OPERATIONS["19 Operation Types"]
            direction TB
            OP_COUNT["count_rows"]
            OP_GB_COUNT["groupby_count"]
            OP_GB_AGG["groupby_agg"]
            OP_FILTER["filter_show"]
            OP_VALUE["value_counts"]
            OP_DESC["describe"]
            OP_CORR["correlation"]
            OP_CROSS["crosstab"]
            OP_PIVOT["pivot_table"]
            OP_CHART["chart"]
            OP_DUP["duplicate_check"]
            OP_NULL["null_analysis"]
            OP_UNIQUE["unique_values"]
            OP_TOP["top_bottom"]
            OP_PCT["percentage"]
            OP_ROLL["rolling_window"]
            OP_CUM["cumulative"]
            OP_RANK["rank"]
            OP_DATE["date_range_analysis"]
        end

        subgraph CHART_GEN["Chart Generation (if operation=chart)"]
            direction TB
            CHART_SPEC["Chart Specification<br/>(type, x, y, groupby, title)"]
            AGG_DATA["Aggregate Data<br/>for Charting"]
            PLOTLY["Generate Plotly Figure<br/>(bar, line, scatter, pie,<br/>histogram, box, area,<br/>heatmap, stacked, grouped)"]
            FIGURE["Plotly Figure Object"]
        end

        RESULT_MD["Markdown Summary<br/>(counts, totals, labels)"]
        STRUCT_RESULT["Structured Result Dict<br/>(operation, result data,<br/>metadata)"]
    end

    subgraph STEP_4["STEP 4: EXPLANATION"]
        direction TB
        CHECK_EXPLAIN{"Has Data<br/>Content?"}
        SKIP_EXPLAIN["Skip Explanation<br/>(general questions,<br/>errors)"]

        subgraph EXPLAIN_PROMPT["Explainer Prompt"]
            direction TB
            EX_QUESTION["User Question"]
            EX_CONV["Conversation Context"]
            EX_RESULT["Structured Result"]
            EX_CUSTOM["Custom Instructions<br/>(terminology, formatting)"]
        end

        BEDROCK_EXPLAIN["AWS Bedrock API Call<br/>Claude Sonnet/Opus 4.5<br/>System: EXPLAINER_PROMPT<br/>Temperature: 0.2"]
        EXPLANATION["Natural Language<br/>Explanation<br/>(3-6 bullet points)"]
    end

    subgraph STEP_5["STEP 5: CONTEXT UPDATE"]
        direction TB
        RECORD["Record Exchange in<br/>Conversation History"]
        SAVE_PLAN["Save: Plan"]
        SAVE_SUMMARY["Save: Result Summary"]
        SAVE_DATE["Save: Date Range"]
        SAVE_FILTERS["Save: Filters Used"]
        SAVE_COLS["Save: Columns Used"]
    end

    subgraph STEP_6["STEP 6: DISPLAY"]
        direction TB
        BUILD_MSG["Build Assistant Message"]
        SHOW_TEXT["Display Markdown Text<br/>(summary + explanation)"]
        SHOW_TABLE{"Has Table<br/>Data?"}
        RENDER_TABLE["Render st.dataframe()<br/>(up to 200 rows)"]
        SHOW_CHART{"Has Chart<br/>Figure?"}
        RENDER_CHART["Render st.plotly_chart()"]
        SHOW_META["Display Metadata<br/>(model used, date range,<br/>instructions applied)"]
        RERUN["st.rerun()<br/>Refresh UI"]
    end

    %% Main flow
    START --> Q_INPUT
    Q_INPUT --> ADD_HISTORY
    ADD_HISTORY --> CALL_ANALYZER

    %% Step 2: Planning
    CALL_ANALYZER --> CHECK_GENERAL
    CHECK_GENERAL -->|Yes| GENERAL_ANSWER
    CHECK_GENERAL -->|No| CHECK_DATA
    CHECK_DATA -->|No| NO_DATA
    CHECK_DATA -->|Yes| PROMPT_ASSEMBLY

    SCHEMA --> BEDROCK_PLAN
    DATE_CTX --> BEDROCK_PLAN
    FEW_SHOT_EX --> BEDROCK_PLAN
    META_PROMPTS --> BEDROCK_PLAN
    KB_SEARCH --> BEDROCK_PLAN
    CUSTOM_INSTR --> BEDROCK_PLAN
    CONV_CTX --> BEDROCK_PLAN
    USER_Q --> BEDROCK_PLAN

    BEDROCK_PLAN --> PARSE_JSON
    PARSE_JSON --> JSON_PLAN

    %% Step 3: Execution
    JSON_PLAN --> RESOLVE_ALIASES
    RESOLVE_ALIASES --> APPLY_FILTERS
    APPLY_FILTERS --> RESOLVE_VALUES
    RESOLVE_VALUES --> FILTERED_DF
    APPLY_FILTERS --> DATE_RANGE
    FILTERED_DF --> DISPATCH

    DISPATCH --> OPERATIONS
    DISPATCH -->|chart| CHART_GEN

    CHART_SPEC --> AGG_DATA
    AGG_DATA --> PLOTLY
    PLOTLY --> FIGURE

    OPERATIONS --> RESULT_MD
    OPERATIONS --> STRUCT_RESULT
    CHART_GEN --> RESULT_MD
    CHART_GEN --> STRUCT_RESULT

    %% Step 4: Explanation
    STRUCT_RESULT --> CHECK_EXPLAIN
    CHECK_EXPLAIN -->|No data| SKIP_EXPLAIN
    CHECK_EXPLAIN -->|Yes| EXPLAIN_PROMPT

    EX_QUESTION --> BEDROCK_EXPLAIN
    EX_CONV --> BEDROCK_EXPLAIN
    EX_RESULT --> BEDROCK_EXPLAIN
    EX_CUSTOM --> BEDROCK_EXPLAIN

    BEDROCK_EXPLAIN --> EXPLANATION

    %% Step 5: Context Update
    EXPLANATION --> RECORD
    RESULT_MD --> RECORD
    RECORD --> SAVE_PLAN
    RECORD --> SAVE_SUMMARY
    RECORD --> SAVE_DATE
    RECORD --> SAVE_FILTERS
    RECORD --> SAVE_COLS

    %% Step 6: Display
    RECORD --> BUILD_MSG
    GENERAL_ANSWER --> BUILD_MSG
    NO_DATA --> BUILD_MSG
    SKIP_EXPLAIN --> BUILD_MSG

    BUILD_MSG --> SHOW_TEXT
    SHOW_TEXT --> SHOW_TABLE
    SHOW_TABLE -->|Yes| RENDER_TABLE
    SHOW_TABLE -->|No| SHOW_CHART
    RENDER_TABLE --> SHOW_CHART
    SHOW_CHART -->|Yes| RENDER_CHART
    SHOW_CHART -->|No| SHOW_META
    RENDER_CHART --> SHOW_META
    SHOW_META --> RERUN

    %% Styling
    classDef inputStyle fill:#3498DB,stroke:#2471A3,color:#FFFFFF,stroke-width:2px
    classDef planStyle fill:#E67E22,stroke:#A85C15,color:#FFFFFF,stroke-width:2px
    classDef execStyle fill:#27AE60,stroke:#1B7A43,color:#FFFFFF,stroke-width:2px
    classDef explainStyle fill:#8E44AD,stroke:#6C3483,color:#FFFFFF,stroke-width:2px
    classDef contextStyle fill:#F39C12,stroke:#B77D0E,color:#FFFFFF,stroke-width:2px
    classDef displayStyle fill:#E74C3C,stroke:#A93226,color:#FFFFFF,stroke-width:2px
    classDef apiStyle fill:#1ABC9C,stroke:#148F77,color:#FFFFFF,stroke-width:2px
    classDef decisionStyle fill:#95A5A6,stroke:#707B7C,color:#FFFFFF,stroke-width:2px

    class START,Q_INPUT,ADD_HISTORY,CALL_ANALYZER inputStyle
    class SCHEMA,DATE_CTX,FEW_SHOT_EX,META_PROMPTS,KB_SEARCH,CUSTOM_INSTR,CONV_CTX,USER_Q,PARSE_JSON,JSON_PLAN planStyle
    class RESOLVE_ALIASES,APPLY_FILTERS,RESOLVE_VALUES,FILTERED_DF,DATE_RANGE,RESULT_MD,STRUCT_RESULT execStyle
    class OP_COUNT,OP_GB_COUNT,OP_GB_AGG,OP_FILTER,OP_VALUE,OP_DESC,OP_CORR,OP_CROSS,OP_PIVOT,OP_CHART,OP_DUP,OP_NULL,OP_UNIQUE,OP_TOP,OP_PCT,OP_ROLL,OP_CUM,OP_RANK,OP_DATE execStyle
    class CHART_SPEC,AGG_DATA,PLOTLY,FIGURE execStyle
    class EX_QUESTION,EX_CONV,EX_RESULT,EX_CUSTOM,EXPLANATION explainStyle
    class RECORD,SAVE_PLAN,SAVE_SUMMARY,SAVE_DATE,SAVE_FILTERS,SAVE_COLS contextStyle
    class BUILD_MSG,SHOW_TEXT,RENDER_TABLE,RENDER_CHART,SHOW_META,RERUN displayStyle
    class BEDROCK_PLAN,BEDROCK_EXPLAIN apiStyle
    class CHECK_GENERAL,CHECK_DATA,DISPATCH,CHECK_EXPLAIN,SHOW_TABLE,SHOW_CHART decisionStyle
    class GENERAL_ANSWER,NO_DATA,SKIP_EXPLAIN decisionStyle
````

## Architecture and Flow Diagrams

````mermaid
graph TB
    subgraph USER_INTERFACE["USER INTERFACE LAYER"]
        direction TB
        APP["app.py<br/>Entry Point<br/>Page Config + Layout"]
        UI["ui_components.py<br/>Streamlit UI Rendering"]
        
        subgraph UI_SECTIONS["UI Sections"]
            SIDEBAR["Sidebar<br/>Data Loading<br/>Dataset Info<br/>Utility Controls"]
            CONFIG_PANEL["Configuration Panel"]
            CHAT["Chat Interface<br/>Message History<br/>Tables + Charts"]
        end
        
        subgraph CONFIG_TABS["Configuration Tabs"]
            TAB_MODEL["Model Selection<br/>Claude Sonnet 4.5<br/>Claude Opus 4.5"]
            TAB_INSTR["Custom Instructions<br/>Context / Rules<br/>Aliases / Mappings"]
            TAB_KB["Knowledge Base<br/>Domain Facts<br/>Definitions"]
            TAB_TRAIN["Training<br/>Few-Shot Examples<br/>Meta-Prompts"]
        end
    end

    subgraph ORCHESTRATION["ORCHESTRATION LAYER"]
        ANALYZER["analyzer.py<br/>Top-Level Orchestrator<br/>Owns DataFrame + Schema<br/>Coordinates Pipeline"]
        CONVERSATION["conversation.py<br/>Conversation Context<br/>Rolling History Window<br/>Follow-Up Resolution"]
    end

    subgraph PIPELINE["ANALYSIS PIPELINE"]
        direction LR
        PLANNER["planner.py<br/>Question to JSON Plan<br/>Schema-Aware Planning<br/>Alias Resolution"]
        EXECUTOR["executor.py<br/>Plan Execution Engine<br/>19 Operation Types<br/>Filter Application"]
        EXPLAINER["explainer.py<br/>Result Summarisation<br/>Natural Language Output<br/>Context-Aware"]
    end

    subgraph SUPPORT["SUPPORT MODULES"]
        direction TB
        CHART["chart_engine.py<br/>Plotly Chart Generation<br/>10 Chart Types<br/>Auto-Aggregation"]
        UTILS["utils.py<br/>JSON Helpers<br/>Schema Extraction<br/>Date Parsing<br/>Filter Engine<br/>General Questions"]
        DATA_LOADER["data_loader.py<br/>CSV Loading<br/>S3 Integration<br/>Encoding Detection"]
        BEDROCK["bedrock_client.py<br/>AWS Bedrock Wrapper<br/>Converse API<br/>Error Handling"]
    end

    subgraph DATA_MODELS["DATA STRUCTURES"]
        direction TB
        DM_CI["CustomInstructions<br/>Context + Rules<br/>Aliases + Mappings"]
        DM_FS["FewShotExample<br/>Question + Plan Pairs<br/>Category Tags"]
        DM_KB["KnowledgeBaseEntry<br/>Title + Content<br/>Tags + Category"]
    end

    subgraph KNOWLEDGE["KNOWLEDGE LAYER"]
        direction TB
        FEW_SHOT["few_shot.py<br/>8 Built-in Examples<br/>8 Meta-Prompts<br/>Custom Examples"]
        KB["knowledge_base.py<br/>In-Memory Store<br/>Keyword Search<br/>Prompt Injection"]
    end

    subgraph CONFIG_LAYER["CONFIGURATION"]
        CONFIG["config.py<br/>Model Definitions<br/>System Prompts<br/>Display Limits<br/>Chart Settings"]
    end

    subgraph EXTERNAL["EXTERNAL SERVICES"]
        direction TB
        BEDROCK_API["AWS Bedrock<br/>Claude Sonnet 4.5<br/>Claude Opus 4.5<br/>Converse API"]
        S3["Amazon S3<br/>CSV File Storage"]
        LOCAL["Local File System<br/>CSV Upload"]
    end

    subgraph DATA_STORE["RUNTIME DATA"]
        SESSION["Streamlit Session State<br/>Chat History<br/>Loaded DataFrame<br/>Instructions<br/>Model Selection"]
        DF["pandas DataFrame<br/>Full Dataset in Memory<br/>Schema Summary"]
    end

    %% UI Layer connections
    APP --> UI
    UI --> SIDEBAR
    UI --> CONFIG_PANEL
    UI --> CHAT
    CONFIG_PANEL --> TAB_MODEL
    CONFIG_PANEL --> TAB_INSTR
    CONFIG_PANEL --> TAB_KB
    CONFIG_PANEL --> TAB_TRAIN

    %% UI to Orchestration
    CHAT --> ANALYZER
    SIDEBAR --> ANALYZER
    TAB_MODEL --> ANALYZER
    TAB_INSTR --> ANALYZER
    TAB_KB --> ANALYZER
    TAB_TRAIN --> ANALYZER

    %% Orchestration to Pipeline
    ANALYZER --> PLANNER
    ANALYZER --> EXECUTOR
    ANALYZER --> EXPLAINER
    ANALYZER --> CONVERSATION

    %% Pipeline to Support
    PLANNER --> BEDROCK
    EXPLAINER --> BEDROCK
    EXECUTOR --> CHART
    EXECUTOR --> UTILS
    PLANNER --> UTILS
    PLANNER --> FEW_SHOT
    PLANNER --> KB

    %% Support to External
    BEDROCK --> BEDROCK_API
    DATA_LOADER --> S3
    DATA_LOADER --> LOCAL

    %% Data flow
    ANALYZER --> DF
    ANALYZER --> SESSION
    DATA_LOADER --> DF

    %% Config connections
    CONFIG --> ANALYZER
    CONFIG --> BEDROCK
    CONFIG --> PLANNER
    CONFIG --> EXECUTOR
    CONFIG --> EXPLAINER

    %% Data models
    DM_CI --> PLANNER
    DM_CI --> EXECUTOR
    DM_CI --> EXPLAINER
    DM_FS --> FEW_SHOT
    DM_KB --> KB

    %% Styling
    classDef uiStyle fill:#4A90D9,stroke:#2C5F8A,color:#FFFFFF,stroke-width:2px
    classDef pipelineStyle fill:#E67E22,stroke:#A85C15,color:#FFFFFF,stroke-width:2px
    classDef supportStyle fill:#27AE60,stroke:#1B7A43,color:#FFFFFF,stroke-width:2px
    classDef externalStyle fill:#8E44AD,stroke:#6C3483,color:#FFFFFF,stroke-width:2px
    classDef dataStyle fill:#E74C3C,stroke:#A93226,color:#FFFFFF,stroke-width:2px
    classDef configStyle fill:#F39C12,stroke:#B77D0E,color:#FFFFFF,stroke-width:2px
    classDef knowledgeStyle fill:#1ABC9C,stroke:#148F77,color:#FFFFFF,stroke-width:2px

    class APP,UI,SIDEBAR,CONFIG_PANEL,CHAT,TAB_MODEL,TAB_INSTR,TAB_KB,TAB_TRAIN uiStyle
    class PLANNER,EXECUTOR,EXPLAINER,ANALYZER,CONVERSATION pipelineStyle
    class CHART,UTILS,DATA_LOADER,BEDROCK supportStyle
    class BEDROCK_API,S3,LOCAL externalStyle
    class SESSION,DF,DM_CI,DM_FS,DM_KB dataStyle
    class CONFIG configStyle
    class FEW_SHOT,KB knowledgeStyle

````
## Project Structure

````tree
.
├── app.py                  # Entry point -- page config, CSS injection, layout composition
├── config.py               # Constants, model IDs, system prompts, display limits
├── data_models.py          # Dataclasses -- CustomInstructions, FewShotExample, KnowledgeBaseEntry
├── bedrock_client.py       # AWS Bedrock converse API wrapper with debug logging
├── data_loader.py          # CSV loading from local file uploads and Amazon S3
├── utils.py                # JSON helpers, schema extraction, date parsing, filter engine
├── conversation.py         # Rolling conversation context for follow-up question resolution
├── knowledge_base.py       # In-memory domain knowledge store with keyword search
├── few_shot.py             # Few-shot examples and meta-prompt instructions manager
├── planner.py              # Natural-language question to JSON execution plan (via Bedrock)
├── executor.py             # JSON plan execution engine -- 19 operation types
├── chart_engine.py         # Plotly chart generation -- 10 chart types
├── explainer.py            # Structured result to natural-language summary (via Bedrock)
├── analyzer.py             # Top-level orchestrator -- owns DataFrame and coordinates pipeline
├── ui_components.py        # Streamlit UI rendering -- sidebar, config panel, chat interface
├── test_bedrock.py         # Bedrock connectivity and model access test script
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation (this file)
````

## File Descriptions

### Entry Point

| File | Lines | Description |
| --- | --- | --- |
| `app.py` | ~40 | Streamlit entry point. Sets page config, injects CSS, initialises session state, and composes the three-region layout: **sidebar**, **configuration panel**, and **chat interface**. Run with `streamlit run app.py`. |

### Configuration and Data Structures

| File | Lines | Description |
| --- | --- | --- |
| `config.py` | ~160 | Central configuration. Stores AWS region, model IDs (*Claude Sonnet 4.5*, *Opus 4.5*), system prompts for the planner and explainer, file size limits, chart settings, and colour palettes. Removes any proxy environment variables on import. |
| `data_models.py` | ~170 | Python dataclasses that hold structured data. `CustomInstructions` stores user context, aliases, value mappings, business rules, terminology, and formatting preferences. `FewShotExample` pairs a question with an expected JSON plan. `KnowledgeBaseEntry` stores domain facts with tags. |

### AWS Integration

| File | Lines | Description |
| --- | --- | --- |
| `bedrock_client.py` | ~70 | Thin wrapper around the boto3 Bedrock Runtime `converse()` API. Creates one client and reuses it. Sends only `temperature` (*not* `top_p`) to satisfy Claude 4.5 constraints. Includes debug logging with **[BedrockClient]** prefix to the terminal for troubleshooting. |
| `data_loader.py` | ~80 | Loads CSV files from local uploads (via Streamlit file uploader) or **Amazon S3**. Tries multiple encodings: *UTF-8*, *Latin-1*, *ISO-8859-1*, *CP1252*. Auto-detects and converts date columns to `datetime` dtype. The `S3Helper` class handles URI parsing, metadata retrieval, and downloading. |

### Analysis Pipeline

| File | Lines | Description |
| --- | --- | --- |
| `planner.py` | ~100 | **First pipeline stage.** Assembles a prompt from the DataFrame schema, few-shot examples, meta-prompts, knowledge base matches, custom instructions, conversation history, and the user question. Calls Claude via Bedrock and parses the JSON response into an execution plan. Handles general questions locally *without* an API call. |
| `executor.py` | ~600 | **Second pipeline stage.** Takes a JSON plan and runs it against the pandas DataFrame. Implements **19 operation types**: `count_rows`, `groupby_count`, `groupby_agg`, `filter_show`, `value_counts`, `describe`, `correlation`, `crosstab`, `pivot_table`, `chart`, `duplicate_check`, `null_analysis`, `unique_values`, `top_bottom`, `percentage`, `rolling_window`, `cumulative`, `rank`, and `date_range_analysis`. Resolves column aliases and value mappings before execution. Delegates chart operations to `chart_engine.py`. |
| `explainer.py` | ~70 | **Third pipeline stage.** Takes the structured result from the executor and asks Claude to produce a **3-6 bullet point** natural-language summary. Includes conversation context and custom instructions (terminology, formatting) in the prompt. Returns an empty string for general questions or errors. |

### Support Modules

| File | Lines | Description |
| --- | --- | --- |
| `utils.py` | ~300 | Shared utility functions. `safe_json_dumps()` handles *numpy/pandas* type serialisation. `df_schema_summary()` extracts column names, dtypes, null counts, unique counts, and example values. `detect_date_columns()` uses a parsing heuristic. `calculate_date_from_relative()` resolves phrases like *"last 30 days"*. `detect_general_question()` matches regex patterns for time, date, greetings. `apply_filters()` supports **12 filter operators**. |
| `chart_engine.py` | ~250 | Generates interactive **Plotly** charts. Supports **10 chart types**: `bar`, `line`, `scatter`, `pie`, `histogram`, `heatmap`, `box`, `area`, `stacked_bar`, and `grouped_bar`. Applies a consistent visual theme with custom colours, *Inter* font, and standardised layout. Auto-aggregates data before charting when needed. |
| `conversation.py` | ~80 | Manages a rolling window of recent question-answer exchanges (default: **last 10**, displays last 5 in prompts). Each exchange stores the question, JSON plan, result summary, date range, filters, and columns used. Generates a formatted context string injected into the planner prompt for **follow-up question resolution**. |

### Knowledge Layer

| File | Lines | Description |
| --- | --- | --- |
| `few_shot.py` | ~180 | Manages few-shot training examples and meta-prompt instructions. Includes **8 built-in examples** covering `groupby`, `filter`, `aggregation`, `chart`, `date`, `pie`, `pivot`, and `value_counts` operations. Includes **8 built-in meta-prompts** that guide the model's reasoning patterns. Users can add custom examples and meta-prompts through the UI. All examples are formatted into the planner prompt for **in-context learning**. |
| `knowledge_base.py` | ~100 | In-memory knowledge store. Entries have a *title*, *content*, *category*, and *tags*. Supports keyword search that scores entries by query-word overlap. The **top 3** matching entries are injected into the planner prompt so the model has domain context when generating plans. |

### Orchestration

| File | Lines | Description |
| --- | --- | --- |
| `analyzer.py` | ~120 | Top-level orchestrator. Owns the DataFrame, schema, conversation context, custom instructions, few-shot manager, and knowledge base. Wires together the planner, executor, and explainer into a single `process_question()` method. Handles **model switching**, **data loading**, and **context updates**. |

### User Interface

| File | Lines | Description |
| --- | --- | --- |
| `ui_components.py` | ~700 | All Streamlit UI rendering. Defines `CUSTOM_CSS` for styling. Implements `initialize_session_state()`, `render_sidebar()` for data loading and dataset info, `render_configuration_panel()` with **4 tabs** (Model, Instructions, Knowledge Base, Training), and `render_chat()` for message history, tables, charts, and input box. Includes **full error handling** that shows errors as chat messages instead of crashing. |

### Testing

| File | Lines | Description |
| --- | --- | --- |
| `test_bedrock.py` | ~80 | Standalone test script for verifying Bedrock connectivity. Tests multiple model IDs with a simple *"Say hello"* prompt. Reports **SUCCESS** or **FAILED** for each model with the response text or error message. Run with `python test_bedrock.py`. |

---

## Installation

### Prerequisites

| Requirement | Details |
| --- | --- |
| **Compute Environment** | AWS SageMaker Studio notebook instance or any environment with AWS credentials |
| **Python Version** | `3.10` or later |
| **IAM Permissions** | `bedrock:InvokeModel` on Claude model ARNs |
| **Bedrock Models** | *Claude Sonnet 4.5* and/or *Claude Opus 4.5* enabled in the [AWS Bedrock console](https://console.aws.amazon.com/bedrock/) |
| **AWS Region** | `us-east-1` (configurable in `config.py`) |

### Step-by-Step Setup

#### 1. Open a SageMaker Studio terminal

Navigate to your workspace directory:

```bash
mkdir -p ~/chatbot
cd ~/chatbot
````

**2. Install Python dependencies**

```python
pip install streamlit pandas numpy boto3 python-dateutil plotly
```
