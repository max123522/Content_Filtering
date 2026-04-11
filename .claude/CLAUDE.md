# 🤖 Claude Project Skills & Style Guide - Semantic DLP Safeguard

## 🎯 Strategic Intent
You are building a high-security Semantic DLP system for **IAI (Israel Aerospace Industries)**. Precision, security, and adherence to corporate branding are the top priorities.

---

## 🛠 Tech Stack Standards
- **LangGraph:** Implement modular, state-driven logic. Every node must have clear input/output types.
- **PydanticAI:** Strict JSON schemas for all LLM outputs. No raw-string reasoning without validation.
- **Branding:** **CRITICAL:** Dynamically extract colors and styles from `https://www.iai.co.il`. Do not hardcode colors unless they match the extracted brand identity.
- **Multilingualism:** Treat Hebrew and English as first-class citizens. Always use **UTF-8** encoding to prevent Hebrew character corruption.

---

## 💻 Coding Standards (PEP 8 & Beyond)
- **Typing:** Use Python Type Hints for all function signatures and class attributes.
- **Documentation:** Follow **PEP 257**. Every public function must have a docstring explaining the logic (especially for complex RAG/Semantic flows).
- **Error Handling:** Use exceptions thoughtfully; catch only what you can handle. Use custom exceptions for DLP-specific errors (e.g., `ExtractionError`).
- **Naming:** Follow `snake_case` for functions/variables and `PascalCase` for classes.

---

## 📦 Dependency & Env Management
- Use a virtual environment (`.venv`).
- All dependencies must be pinned in `requirements.txt`.
- **Security:** Sensitive data (API Keys, DB URLs) must ONLY be accessed via `config.py` using environment variables.

---

## 🚀 Build & Test Commands
- **Setup Environment:** `pip install -r requirements.txt`
- **Ingest Initial Data:** `python data/terms_loader.py` (Parses the provided `.docx` file).
- **Launch Backend:** `python main.py`
- **Launch UI:** `python app_flask.py`
- **Reset Database:** `python data/db_handler.py --reset`

---

## ⚠️ Critical Rules
1. **Semantic over Keyword:** Never rely on simple string matching. Always prioritize vector similarity and LLM context analysis for both Hebrew and English.
2. **Morphology:** Account for Hebrew prefixes (ו, ה, ב, ל) when performing initial term detection before LLM reasoning.
3. **Isolation:** Do not attempt to reach external APIs not defined in `config.py`. 
4. **Clean Workspace:** Keep notebooks for R&D only; production logic stays in `.py` files within the defined directory structure.