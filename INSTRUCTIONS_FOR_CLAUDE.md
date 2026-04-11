
# 🛡️System Design Spec: Semantic DLP Safeguard (IAI Edition)
## 1. Project Vision & Mission
You are acting as a Senior AI Solutions Architect.
The mission is to build a high-end Semantic Data Loss Prevention (DLP) system for IAI (Israel Aerospace Industries).
The Objective:
Automate the "Sanitization" (White-washing) process for sensitive documents (PDF, Word, Docx, Doc).
The system must decide if a document is safe to be exported or if it contains classified information based on a dynamic list of terms.

The Innovation:
Standard DLP tools use "Keyword Matching" which causes high False-Alarm rates.
Our system uses Deep Semantic Reasoning to distinguish context:
Casual Context: "The pilot adjusted the Arrow on the display." (Approve)
Classified Context: "Internal telemetry from the Arrow-3 interceptor." (Block)


## 2. Technical Stack & Architecture
Core Technologies:
Orchestration: LangGraph (To manage the state-machine of the analysis).
Structured AI: PydanticAI (To ensure the LLM returns a strict JSON: Decision, Reasoning, Confidence).
Local Embeddings: Snowflake Arctic Embed v2.0 (Must run locally for security).
Validation: BGE-Reranker (To cross-check the LLM's decision against the DB description).
Database: SQLite for Development, Oracle for Production.
Hybrid Environment Support (Crucial):
The system must use a config.py to toggle between:
DEV (Laptop): Use OpenAI API (GPT-4o/Claude-3.5) with a personal API Key.
PROD (IAI On-Prem): Use internal endpoint http://internal-iai-gpt-oss-12b with api_key="na".

## 3. Knowledge Injection (RAG-on-Terms)
The LLM does not know internal IAI projects (e.g., "Lampad","Arrow").
You must implement a Knowledge Injection flow:
Detection: Identify a forbidden term from the DB in the text.
Context Retrieval: Fetch the forbidden_context_description associated with that term from the DB.
Prompt Enrichment: Inject that description into the LLM's system prompt so it understands the specific classified nature of the term it just found.

###  Multilingual & Fuzzy Semantic Awareness (CRITICAL)
Terms in the Word file are in English, but target docs are in **Hebrew/English**. 
1. **Fuzzy & Variant Matching:** The system MUST detect terms despite spelling variations, typos, or linguistic inflections. This includes Hebrew variations (e.g., 'חיזבאללה' vs 'חיזבלה') and English variants (e.g., 'Hezbollah' vs 'Hizbala').
2. **Semantic Vector Search:** Use the Embedding model (Snowflake Arctic) to identify that a Hebrew variation maps to the same forbidden concept as the English term in the DB. The system should look for "Meaning" rather than "Characters".
3. **Morphological Handling:** The system must handle Hebrew prefixes (e.g., 'והחיזבאללה', 'למפאד') and English plurals/tenses by comparing their semantic embeddings.
4. **Cross-Lingual Reasoning:** The LLM receives the DB context (English) and the segment (Hebrew/English) and decides if they refer to the same classified entity.


## 4. UI/UX & Branding (IAI Corporate Identity)
The Flask GUI (Chatbot & Admin Dashboard) must be Enterprise-Grade.
Visual Crawling: Use your browsing capabilities to visit the official IAI website (https://www.iai.co.il).
Brand Extraction: Extract primary/secondary hex colors, typography, and button styles.
Implementation: Save these as CSS variables in static/css/brand_identity.css.
Design Style: Use a modern "Aerospace/High-Tech" look. Implement a clean layout with Deep Blues, Metallic Silvers, and Glassmorphism effects where appropriate.
Admin & History : - Admin panel to manage terms.
Audit Log Page: A dedicated view to browse all past scans, see why they were blocked, and review the metadata.

Components:
Modern Drag & Drop: For document uploads.
Live Result Dashboard: Color-coded status (Red for Blocked, Green for Approved) with IAI-themed typography.
Admin Panel: Secure dashboard to manage forbidden_terms and edit context_descriptions.
Feedback Loop: "Like/Dislike" buttons on results.


## 5. Database & Logging (Audit Trail)
CRITICAL: The db_handler.py must implement an analysis_logs table to ensure transparency and accountability.

### Table A: forbidden_terms (Knowledge Base)
id (Primary Key)
term (The forbidden word/phrase)

context_description (Classified context for the LLM to understand the term's sensitive nature)

### TABLE B: analysis_logs
id (Primary Key)
timestamp (When the scan happened)
filename (Name of the uploaded document)
decision (Approved OR Blocked)
reasoning (The full semantic explanation provided by the LLM)
confidence_score (The model's certainty level)

Persistence: Every single scan attempt must be persisted to this table before the result is shown to the user.

## 6. Project Structure (File System)
The project must follow this modular structure to ensure clean separation:

Plaintext
IAI_DLP_PROJECT/
│
├── config.py               # Environment variables, API Keys, Model URLs (DEV/PROD toggle)
├── main.py                 # Entry point to run the Backend/LangGraph logic
├── app_flask.py            # High-end Flask Web Server (Frontend & Admin)
│
├── core/                   # Logic Layer
│   ├── graph_manager.py    # LangGraph Workflow definition
│   ├── agents.py           # PydanticAI agents for reasoning
│   └── document_parser.py  # PDF/Docx extraction (PyMuPDF / python-docx)
│
├── data/                   # Data Layer
│   ├── db_handler.py       # Implements analysis_logs schema
│   └── terms_loader.py     # Script to ingest the initial Word file
│
├── models/                 # AI Model Wrappers
│   ├── embedding_service.py # Snowflake Arctic local implementation
│   └── reranker_service.py  # BGE-Reranker implementation
│
├── static/                 # CSS/JS and IAI Logo
└── templates/              # HTML files for the High-End GUI


## 7. Smart Data Loading (Delta Logic)
terms_loader.py must parse words_content_filtering.docx (delimited by /).
CRITICAL: Implement "Upsert" logic.
Compare the file with the DB;
only add new terms or update existing ones.
Do NOT overwrite existing context_description fields that were manually enriched in the DB.

## 8. Execution Instructions for Claude Code:
Read this INSTRUCTIONS_FOR_CLAUDE.md entirely.
Initialize the directory structure and config.py.
Build db_handler.py first with the dual-table schema (Terms & Logs).
Run terms_loader.py to sync terms from the Word file.
Build the LangGraph workflow and the Branded Flask GUI (including the History page).