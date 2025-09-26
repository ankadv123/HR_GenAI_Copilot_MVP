# HR GenAI Copilot (Consumer Industry) — MVP

A **Streamlit** app that answers HR policy FAQs, looks up employee info, shows shift schedules/training, and generates HR letters. 
Designed for frontline retail/consumer-workforce needs (store & warehouse).

## 1) Quick Start (No LLM needed)
```bash
# 1. Unzip the project
cd hr_genai_mvp

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```
Your browser will open at `http://localhost:8501`.

## 2) Enable LLM (Optional)
If you want the bot to **paraphrase responses, cite sources, or translate**:
```bash
# Create a .env file (or copy .env.example -> .env) and add your key
OPENAI_API_KEY=sk-...

# Start the LLM-enabled app
streamlit run app_llm.py
```
Notes:
- Uses `gpt-4o-mini` by default (change inside `app_llm.py` if needed).
- The app will still **retrieve** the best matching policy/FAQ locally; the LLM is used to polish/translate the final message and add citations.

## 3) Project Structure
```
hr_genai_mvp/
├─ app.py                # Local, no-LLM app (deterministic demo)
├─ app_llm.py            # Optional LLM-enhanced app
├─ requirements.txt
├─ .env.example
├─ data/
│  ├─ hr_policies.csv
│  ├─ hr_faq.csv
│  ├─ employees.csv
│  ├─ shifts.csv
│  └─ training.csv
└─ README.md
```

## 4) Demo Scenarios (Script)
1) **Leave accrual** — “How many paid leaves do I get per month?”  
2) **Overtime** — “Is overtime paid and at what rate?”  
3) **Shifts & training** — Enter EmpID `C001` in *Shifts & Training* and click **Show Status**.  
4) **Letter** — Generate an **Experience Letter** for employee `C005` and download the `.txt`.

## 5) Next Steps (Roadmap)
- Integrate with HRMS (leave/attendance APIs).
- Add multi-lingual support (Hindi + regional).
- Add authentication (associate vs manager views).
- Add analytics (query volumes, top topics, deflection %).
