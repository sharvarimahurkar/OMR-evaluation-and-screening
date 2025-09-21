# omr_processing.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import datetime

app = FastAPI()

# Allow frontend (Streamlit) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store of results
past_results = []

# Dummy subject distribution
SUBJECTS = ["PYTHON", "DATA ANALYSIS", "MySQL", "POWER BI", "Adv STATS"]

@app.post("/evaluate")
async def evaluate(file: UploadFile, candidate_name: str = Form(None)):
    try:
        # Load uploaded image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # 📝 TODO: Replace with your actual OMR evaluation logic
        # Here we simulate per-subject scores
        import random
        per_subject = {subj: random.randint(10, 20) for subj in SUBJECTS}
        total = sum(per_subject.values())

        # Create a simple "overlay" by drawing a border
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([10, 10, overlay.width - 10, overlay.height - 10], outline="red", width=5)

        buf = BytesIO()
        overlay.save(buf, format="PNG")
        overlay_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")

        result = {
            "candidate_name": candidate_name or "Anonymous",
            "total": total,
            "per_subject": per_subject,
            "overlay_bytes": overlay_bytes,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        past_results.append(result)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/results")
def get_results():
    return past_results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
