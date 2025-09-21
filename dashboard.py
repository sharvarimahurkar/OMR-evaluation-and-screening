# dashboard.py

import streamlit as st
import requests
import pandas as pd
from io import BytesIO
import base64

# Page config
st.set_page_config(page_title="OMR Evaluation Dashboard", layout="wide")
st.title("📊 Automated OMR Evaluation Dashboard")

api_base = "http://localhost:8000"

# --- Upload Form ---
with st.form("upload_form"):
    candidate_name = st.text_input("Candidate name (optional)")
    uploaded_file = st.file_uploader("Upload OMR Sheet", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Evaluate")

if submitted and uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"candidate_name": candidate_name}
    try:
        resp = requests.post(f"{api_base}/evaluate", files=files, data=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        # --- Overall score ---
        st.subheader("✅ Overall Performance")
        st.metric("Final Score", result["total"])

        # --- Subject-wise analysis ---
        st.subheader("📚 Subject-wise Scores")
        per_subject = result["per_subject"]

        # Each subject has 20 questions
        subject_totals = {
            "PYTHON": 20,
            "DATA ANALYSIS": 20,
            "MySQL": 20,
            "POWER BI": 20,
            "Adv STATS": 20,
        }

        # DataFrame with Correct / Total
        df_subject = pd.DataFrame(
            [(subj, f"{score}/{subject_totals[subj]}") for subj, score in per_subject.items()],
            columns=["Subject", "Score"]
        )

        st.table(df_subject)
        st.bar_chart(
            pd.DataFrame(
                per_subject.values(),
                index=per_subject.keys(),
                columns=["Correct"]
            )
        )

        # --- Overlay OMR result ---
        st.subheader("📝 Answer Sheet Overlay")
        overlay_bytes = base64.b64decode(result["overlay_bytes"])
        st.image(BytesIO(overlay_bytes), caption=f"Evaluated OMR — {result.get('candidate_name')}", use_column_width=True)

    except Exception as e:
        st.error(f"Error contacting backend: {e}")

# --- Past Results Section ---
st.markdown("---")
if st.button("Load past results"):
    try:
        r = requests.get(f"{api_base}/results", timeout=10)
        r.raise_for_status()
        rows = r.json()
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
        else:
            st.info("No past results found.")
    except Exception as e:
        st.error(f"Failed to fetch results: {e}")
