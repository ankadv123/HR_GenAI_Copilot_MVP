# app_gemini.py
# Streamlit HR GenAI Copilot (Consumer, HR)
# RAG: sentence-transformers (MiniLM)
# LLM polish/answer: Google Gemini (if GOOGLE_API_KEY is set), else FLAN-T5 (local, free)
# Data: CSVs in ./data. Generate large datasets with: python seed_data.py

import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# RAG & local LLM
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Gemini (optional)
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai

# -------------------- Config --------------------
st.set_page_config(page_title="HR GenAI Copilot", page_icon="🛍️", layout="wide")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

GEMINI_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
    except Exception as e:
        st.sidebar.warning(f"Could not configure Gemini API: {e}")

# -------------------- Data Load --------------------
@st.cache_data
def load_data():
    policies = pd.read_csv(os.path.join(DATA_DIR, "hr_policies.csv"))
    faqs = pd.read_csv(os.path.join(DATA_DIR, "hr_faq.csv"))
    employees = pd.read_csv(os.path.join(DATA_DIR, "employees.csv"))
    shifts = pd.read_csv(os.path.join(DATA_DIR, "shifts.csv"))
    training = pd.read_csv(os.path.join(DATA_DIR, "training.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    tickets_path = os.path.join(DATA_DIR, "tickets.csv")
    if os.path.exists(tickets_path):
        tickets = pd.read_csv(tickets_path)
    else:
        tickets = pd.DataFrame(columns=["ticket_id","created_at","channel","user_input","intent","confidence","status","notes"])
    return policies, faqs, employees, shifts, training, orders, products, tickets

policies, faqs, employees, shifts, training, orders, products, tickets = load_data()

# -------------------- Models (cached) --------------------
@st.cache_resource(show_spinner=True)
def get_rag_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    t5 = pipeline("text2text-generation", model="google/flan-t5-small")
    return embedder, t5

embedder, t5 = get_rag_models()

@st.cache_resource(show_spinner=True)
def get_gemini_model(model_name: str = "gemini-2.5-flash"):
    if not GEMINI_KEY:
        return None
    try:
        return genai.GenerativeModel(model_name)
    except Exception:
        return None

gemini = get_gemini_model()

# -------------------- KB Embeddings --------------------
@st.cache_data(show_spinner=False)
def build_kb_embeddings(faqs_df, policies_df):
    faq_texts = (faqs_df["question"].fillna("") + " || " + faqs_df["answer"].fillna("")).tolist()
    faq_emb = embedder.encode(faq_texts, convert_to_tensor=False, normalize_embeddings=True)

    pol_texts = (policies_df["topic"].fillna("") + " || " + policies_df["content"].fillna("")).tolist()
    pol_emb = embedder.encode(pol_texts, convert_to_tensor=False, normalize_embeddings=True)
    return faq_texts, faq_emb, pol_texts, pol_emb

faq_texts, faq_emb, pol_texts, pol_emb = build_kb_embeddings(faqs, policies)

def top_match(query: str, texts, emb, top_k=1):
    qv = embedder.encode([query], convert_to_tensor=False, normalize_embeddings=True)
    sims = cosine_similarity([qv[0]], emb)[0]
    ranked = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)
    if top_k == 1:
        idx, score = ranked[0]
        return idx, float(score)
    return ranked[:top_k]

# -------------------- Generation Helpers --------------------
def t5_polish(text: str, tag: str, enabled: bool = True) -> str:
    if not enabled:
        return f"{text.strip()} [{tag}]"
    prompt = (
        "Rewrite clearly in simple English under 80 words. Keep facts unchanged. "
        "End with the given tag in square brackets.\n\n"
        f"TEXT:\n{text}\n\nTAG: [{tag}]"
    )
    try:
        out = t5(prompt, max_length=260, do_sample=False)
        ans = out[0]["generated_text"].strip()
        if f"[{tag}]" not in ans:
            ans = f"{ans} [{tag}]"
        return ans
    except Exception:
        return f"{text.strip()} [{tag}]"

def gemini_polish(text: str, tag: str) -> str:
    if gemini is None:
        return f"{text.strip()} [{tag}]"
    prompt = (
        "Rewrite the following in clear, simple English under 80 words. "
        "Keep facts unchanged. End with the given tag in square brackets.\n\n"
        f"TEXT:\n{text}\n\nTAG: [{tag}]"
    )
    try:
        resp = gemini.generate_content(prompt)
        out = (resp.text or "").strip()
        if f"[{tag}]" not in out:
            out = f"{out} [{tag}]"
        return out
    except Exception:
        return f"{text.strip()} [{tag}]"

def gemini_answer_with_context(query: str, passages: list[str], tag: str) -> str:
    """RAG-style answer composition using retrieved passages (optional)."""
    if gemini is None:
        # fallback: simple concat of top passages
        ctx = "\n\n".join(passages[:2])
        return f"{ctx}\n\n[{tag}]"
    context = "\n\n".join([f"- {p}" for p in passages[:5]])
    prompt = (
        "You are an HR assistant for a consumer retail company. Answer the user's query "
        "STRICTLY using the provided context. If the answer is not present, say you don't have that "
        "info and suggest escalating. Keep under 200 words. End with the given [TAG].\n\n"
        f"Query:\n{query}\n\nContext:\n{context}\n\nTAG: [{tag}]"
    )
    try:
        resp = gemini.generate_content(prompt)
        out = (resp.text or "").strip()
        if f"[{tag}]" not in out:
            out = f"{out} [{tag}]"
        return out
    except Exception:
        return f"Sorry, I couldn't generate an answer. [{tag}]"

# -------------------- Ticketing --------------------
def create_ticket(user_input, intent, confidence, notes=""):
    ticket_id = f"T-{uuid.uuid4().hex[:8].upper()}"
    created_at = datetime.now().isoformat(timespec="seconds")
    row = {"ticket_id":ticket_id,"created_at":created_at,"channel":"app",
           "user_input":user_input,"intent":intent,"confidence":round(float(confidence or 0.0),2),
           "status":"OPEN","notes":notes}
    path = os.path.join(DATA_DIR,"tickets.csv")
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.DataFrame(columns=["ticket_id","created_at","channel","user_input","intent","confidence","status","notes"])
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return ticket_id

# -------------------- UI --------------------
st.title("🛍️ HR GenAI Copilot")
st.caption("RAG with MiniLM + optional Gemini or FLAN-T5 polish. ")

with st.sidebar:
    st.header("🔧 Settings")
    mode = st.radio("Select mode", ["Chatbot Q&A", "Employee Lookup", "Letter Generator", "Customer Support", "Escalations"])
    st.markdown("---")
    use_gemini = st.toggle("Use Gemini (needs GOOGLE_API_KEY)", value=bool(GEMINI_KEY))
    use_t5 = st.toggle("Use FLAN-T5 polish (local, free)", value=not use_gemini)
    st.caption(f"Gemini configured: {bool(GEMINI_KEY)}")
    st.markdown("**Data folder**")
    st.code(DATA_DIR, language="bash")

# -------------------- Modes --------------------
if mode == "Chatbot Q&A":
    st.subheader("💬 Ask an HR question")
    query = st.text_input("e.g., How many paid leaves do I get per month?")

    if st.button("Search") and query.strip():
        i_faq, s_faq = top_match(query, faq_texts, faq_emb)
        i_pol, s_pol = top_match(query, pol_texts, pol_emb)

        if s_faq >= s_pol and s_faq > 0.25:
            qa = faqs.iloc[i_faq]
            raw = qa["answer"]
            # Option A (simple polish)
            if use_gemini:
                answer = gemini_polish(raw, "FAQ")
            else:
                answer = t5_polish(raw, "FAQ", enabled=use_t5)
            # Option B (compose with context):
            # passages = [qa["question"], qa["answer"]]
            # answer = gemini_answer_with_context(query, passages, "FAQ") if use_gemini else t5_polish(raw, "FAQ", enabled=use_t5)
            st.success(answer)
            with st.expander("Matched FAQ"):
                st.write(f"**Q:** {qa['question']}")
                st.caption(f"Similarity: {s_faq:.2f}")
        elif s_pol > 0.25:
            pr = policies.iloc[i_pol]
            raw = pr["content"]
            if use_gemini:
                answer = gemini_polish(raw, pr["topic"])
            else:
                answer = t5_polish(raw, pr["topic"], enabled=use_t5)
            st.success(answer)
            with st.expander("Matched Policy"):
                st.write(f"**Topic:** {pr['topic']}")
                st.caption(f"Similarity: {s_pol:.2f}")
        else:
            st.warning("Low confidence. Escalating to HR...")
            tid = create_ticket(user_input=query, intent="hr_policy", confidence=max(s_faq, s_pol), notes="Low similarity")
            st.info(f"Ticket created: **{tid}**")

elif mode == "Employee Lookup":
    st.subheader("🧑‍💼 Employee Directory")
    q = st.text_input("Search by name, Employee ID, role, location, or manager")
    if st.button("Find Employee") and q.strip():
        mask = (
            employees["emp_id"].astype(str).str.contains(q, case=False) |
            employees["name"].astype(str).str.contains(q, case=False) |
            employees["role"].astype(str).str.contains(q, case=False) |
            employees["location"].astype(str).str.contains(q, case=False) |
            employees["manager"].astype(str).str.contains(q, case=False)
        )
        res = employees[mask]
        if not res.empty:
            st.success(f"Found {len(res)} record(s).")
            st.dataframe(res, use_container_width=True)
        else:
            st.warning("No matching employee found. Creating ticket...")
            tid = create_ticket(user_input=q, intent="employee_lookup", confidence=0.0, notes="Not found")
            st.info(f"Ticket created: **{tid}**")
    st.markdown("### Directory")
    st.dataframe(employees, use_container_width=True)

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
        # Pick polish engine
        if use_gemini:
            final_text = gemini_polish(body, letter_type.upper())
        else:
            final_text = t5_polish(body, tag=letter_type.upper(), enabled=use_t5)

        st.text_area("Letter Preview", final_text, height=300)
        st.download_button("Download .txt", final_text.encode("utf-8"),
                           file_name=f"{letter_type.replace(' ','_')}_{emp_id}.txt")

elif mode == "Customer Support":
    st.subheader("🛒 Customer Support — Orders & Product Availability")
    tab1, tab2 = st.tabs(["Order Status", "Product Availability"])

    with tab1:
        oid = st.text_input("Enter Order ID (e.g., O-2025-1001)")
        if st.button("Track Order"):
            row = orders[orders['order_id']==oid]
            if not row.empty:
                r = row.iloc[0]
                msg = f"Order {r['order_id']} — {r['item']} • {r['status']} • ETA {r['eta']}"
                pretty = gemini_polish(msg, "ORDER") if use_gemini else t5_polish(msg, "ORDER", enabled=use_t5)
                st.success(pretty)
                if isinstance(r.get("carrier", ""), str) and r["carrier"]:
                    st.caption(f"Carrier: {r['carrier']} | Tracking: {r['tracking']}")
            else:
                st.warning("Order not found. Escalating to human support...")
                tid = create_ticket(user_input=f"Order lookup: {oid}", intent="order_status", confidence=0.0, notes="Order ID not found")
                st.info(f"Ticket created: **{tid}**")

    with tab2:
        q = st.text_input("Search by Product name/SKU and City (e.g., 'Wireless Mouse in Mumbai')")
        if st.button("Check Availability"):
            parts = q.split(" in ")
            name = parts[0].strip()
            city = parts[1].strip() if len(parts)>1 else ""
            df = products.copy()
            if name:
                df = df[df['name'].str.contains(name, case=False) | df['sku'].str.contains(name, case=False)]
            if city:
                df = df[df['city'].str.contains(city, case=False)]
            if not df.empty:
                msg = f"Found {len(df)} matching store(s)."
                pretty = gemini_polish(msg, "INVENTORY") if use_gemini else t5_polish(msg, "INVENTORY", enabled=use_t5)
                st.success(pretty)
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No availability found. Escalating to human support...")
                tid = create_ticket(user_input=f"Availability: {q}", intent="product_availability", confidence=0.0, notes="No matches")
                st.info(f"Ticket created: **{tid}**")

elif mode == "Escalations":
    st.subheader("📨 Escalations (Tickets)")
    try:
        df = pd.read_csv(os.path.join(DATA_DIR,"tickets.csv"))
        if df.empty:
            st.info("No tickets yet.")
        else:
            st.dataframe(df.sort_values("created_at", ascending=False), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read tickets: {e}")
