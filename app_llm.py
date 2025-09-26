
import os
import streamlit as st
import pandas as pd
import math, re
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

# Optional: OpenAI rephrasing/translation/citation
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
use_llm = bool(OPENAI_API_KEY)

if use_llm:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        use_llm = False

st.set_page_config(page_title="HR GenAI Copilot (LLM)", page_icon="🤖", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_data():
    policies = pd.read_csv(os.path.join(DATA_DIR, "hr_policies.csv"))
    faqs = pd.read_csv(os.path.join(DATA_DIR, "hr_faq.csv"))
    employees = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"))
    shifts = pd.read_csv(os.path.join(DATA_DIR, "shifts.csv"))
    training = pd.read_csv(os.path.join(DATA_DIR, "training.csv"))
    return policies, faqs, employees, shifts, training

policies, faqs, employees, shifts, training = load_data()

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]

def cosine_sim(a, b):
    wa, wb = Counter(tokenize(a)), Counter(tokenize(b))
    common = set(wa) & set(wb)
    num = sum(wa[t]*wb[t] for t in common)
    da = (sum(v*v for v in wa.values())) ** 0.5
    db = (sum(v*v for v in wb.values())) ** 0.5
    return 0.0 if da*db == 0 else num/(da*db)

def best_match(query, df, text_cols):
    scores = []
    for i, row in df.iterrows():
        corpus = " ".join(str(row[c]) for c in text_cols if c in df.columns)
        scores.append((i, cosine_sim(query, corpus)))
    scores.sort(key=lambda x: x[1], reverse=True)
    if not scores:
        return None, 0.0
    idx, sc = scores[0]
    return df.iloc[idx], sc

def llm_polish(answer_text, source_label, policy_or_faq):
    if not use_llm:
        return answer_text
    sys = "You are an HR assistant for a large consumer retail company. Be concise, accurate, and policy-aligned."
    user = f"""Rewrite the following answer for clarity in plain English. Add a short citation tag like [{policy_or_faq}:{source_label}]. 
If the user query seems to be in Hinglish/Hindi, translate to English first and then answer. Keep it under 80 words.
ANSWER:\n{answer_text}"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return answer_text

st.title("🤖 HR GenAI Copilot — LLM Enhanced")
st.caption("Retrieval + Optional LLM polish/translation. Toggle by setting OPENAI_API_KEY.")

with st.sidebar:
    st.header("🔧 Actions")
    mode = st.radio("Select mode", ["Chatbot Q&A", "Employee Lookup", "Shifts & Training", "Letter Generator"])
    st.markdown("---")
    st.markdown(f"LLM enabled: **{use_llm}**")
    st.markdown("**Data Sources**")
    st.write(DATA_DIR)

if mode == "Chatbot Q&A":
    st.subheader("💬 Ask an HR question")
    query = st.text_input("e.g., How many paid leaves do I get per month? (Hindi/English both ok)")
    if st.button("Search") and query.strip():
        faq_row, faq_score = best_match(query, faqs, ["question","answer"])
        pol_row, pol_score = best_match(query, policies, ["topic","content"])
        if (faq_row is not None and faq_score >= pol_score) and faq_score > 0.1:
            raw = f"{faq_row['answer']}"
            polished = llm_polish(raw, "FAQ", "FAQ")
            st.success(polished)
            with st.expander("Matched FAQ"):
                st.write(f"**Q:** {faq_row['question']}")
                st.caption(f"Similarity: {faq_score:.2f}")
        elif pol_row is not None and pol_score > 0.1:
            raw = f"{pol_row['content']}"
            polished = llm_polish(raw, pol_row['topic'], "POLICY")
            st.success(polished)
            with st.expander("Matched Policy"):
                st.write(f"**Topic:** {pol_row['topic']}")
                st.caption(f"Similarity: {pol_score:.2f}")
        else:
            st.warning("No relevant context found. Please rephrase your question.")

elif mode == "Employee Lookup":
    st.subheader("🧑‍💼 Employee Directory")
    q = st.text_input("Search by name or Employee ID")
    if st.button("Find Employee") and q.strip():
        row, score = best_match(q, employees, ["emp_id","name","role","location","manager"])
        if row is not None and score > 0.1:
            st.success(f"Found: {row['name']} ({row['emp_id']})")
            st.write(pd.DataFrame([row]))
        else:
            st.warning("No matching employee found.")
    st.markdown("### Directory")
    st.dataframe(employees, use_container_width=True)

elif mode == "Shifts & Training":
    st.subheader("📆 Today's Shift & Training")
    today = datetime.now().strftime("%Y-%m-%d")
    st.caption(f"Today: {today}")
    emp_id = st.text_input("Enter Employee ID (e.g., C001)")
    if st.button("Show Status") and emp_id.strip():
        sh = shifts[(shifts['emp_id']==emp_id) & (shifts['date']==today)]
        tr = training[training['emp_id']==emp_id]
        if not sh.empty:
            st.success(f"Shift for {emp_id}: {sh.iloc[0]['shift']} at {sh.iloc[0]['store']}")
        else:
            st.info("No shift found for today.")
        if not tr.empty:
            st.markdown("**Training Status**")
            st.dataframe(tr, use_container_width=True)
        else:
            st.info("No training records.")

elif mode == "Letter Generator":
    st.subheader("📝 Generate HR Letter")
    emp = st.selectbox("Select Employee", employees['emp_id'] + " - " + employees['name'])
    letter_type = st.selectbox("Letter Type", ["Offer Letter", "Experience Letter", "Warning Letter"])
    emp_id = emp.split(" - ")[0]
    emp_row = employees[employees['emp_id']==emp_id].iloc[0]
    today = datetime.now().date().isoformat()

    if st.button("Generate"):
        if letter_type == "Offer Letter":
            body = f"""
Date: {today}

Subject: Offer of Employment

Dear {emp_row['name']},

We are pleased to offer you the position of {emp_row['role']} at our {emp_row['location']} location.
Your start date is {today}. Additional details will be provided during onboarding.

Sincerely,
HR Team
"""
        elif letter_type == "Experience Letter":
            body = f"""
Date: {today}

To Whom It May Concern,

This is to certify that {emp_row['name']} (Emp ID: {emp_id}) worked as {emp_row['role']}
at our {emp_row['location']} store from {emp_row['join_date']} to {today}. During this period,
their conduct and performance were satisfactory.

Sincerely,
HR Team
"""
        else:
            body = f"""
Date: {today}

Subject: Warning Letter – Attendance

Dear {emp_row['name']},

This is a formal warning regarding repeated late attendance. Please treat this as a notice to improve
punctuality as per policy P002. Further occurrences may lead to disciplinary action.

Sincerely,
HR Team
"""
        st.text_area("Letter Preview", body, height=300)
        st.download_button("Download .txt", body.encode("utf-8"), file_name=f"{letter_type.replace(' ','_')}_{emp_id}.txt")
