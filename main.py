# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import sqlite3, base64, io
from datetime import datetime
import cv2
import numpy as np

app = FastAPI()

# Allow Streamlit to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create SQLite DB if not exists
def init_db():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            total INTEGER,
            per_subject TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...), candidate_name: str = Form("")):
    # Read file into OpenCV
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Fake evaluation for demo
    total_score = np.random.randint(0, 100)
    per_subject = {"Math": np.random.randint(0, 25), "Physics": np.random.randint(0, 25), "Chemistry": np.random.randint(0, 25)}

    # Create overlay (just putting text for now)
    overlay = image.copy()
    cv2.putText(overlay, f"Score: {total_score}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    _, buffer = cv2.imencode(".png", overlay)
    overlay_b64 = base64.b64encode(buffer).decode("utf-8")

    # Save result in DB
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("INSERT INTO results (candidate_name, total, per_subject, timestamp) VALUES (?, ?, ?, ?)",
              (candidate_name or "Unknown", total_score, str(per_subject), datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return {
        "candidate_name": candidate_name or "Unknown",
        "total": total_score,
        "per_subject": per_subject,
        "overlay_b64": overlay_b64,
    }

@app.get("/results")
def get_results():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("SELECT id, candidate_name, total, per_subject, timestamp FROM results ORDER BY id DESC")
    rows = [{"id": r[0], "candidate_name": r[1], "total": r[2], "per_subject": r[3], "timestamp": r[4]} for r in c.fetchall()]
    conn.close()
    return rows
