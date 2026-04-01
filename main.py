from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import pandas as pd
from io import StringIO, BytesIO

# Initialize app
app = FastAPI()

# ─── CORS CONFIGURATION ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For now allow all (later restrict to your frontend domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ROOT ENDPOINT ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Enerlytics API running 🚀"}

# ─── HEALTH CHECK (Optional but useful) ────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─── ANALYTICS ENDPOINT ───────────────────────────────────────
@app.get("/analytics")
def analytics():
    return {
        "stats": {
            "totalConsumption": 12000,
            "avgDaily": 400,
            "peakDemand": 250,
            "peakDay": "Monday",
            "estimatedCost": 1500,
            "baseload": 120,
            "daysOfData": 30,
            "trend": {
                "consumptionChange": 5,
                "costChange": 3
            }
        },
        "daily": [
            {"date": "2024-01-01", "label": "Jan 1", "consumption": 400}
        ],
        "weekly": [
            {"week": "Week 1", "consumption": 2800}
        ],
        "monthly": [
            {"month": "Jan", "consumption": 12000}
        ]
    }
# ─── Upload CSV / Excel ───────────────────────────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()

    # Detect file type and parse accordingly
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "csv":
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(BytesIO(contents))
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Please upload a .csv, .xlsx, or .xls file."
        )

    # Identify time columns (all HH:MM columns)
    time_columns = [col for col in df.columns if ":" in col]

    # Convert wide → long format
    df_long = df.melt(
        id_vars=["Date"],
        value_vars=time_columns,
        var_name="time",
        value_name="consumption"
    )

    # Create timestamp column
    df_long["timestamp"] = pd.to_datetime(
        df_long["Date"] + " " + df_long["time"]
    )

    # Clean final dataset
    df_final = df_long[["timestamp", "consumption"]].sort_values("timestamp")

    return {
        "success": True,
        "rowsProcessed": len(df_final),
        "message": "File processed successfully",
        "sample": df_final.head(5).to_dict(orient="records")
    }
    # ─── Anomalies───────────────────────────────────────
@app.get("/anomalies")
def get_anomalies():
    return {
        "anomalies": [
            {
                "date": "2024-01-01",
                "time": "14:00",
                "consumption": 500,
                "expected": 300,
                "deviation": 200,
                "severity": "high",
                "type": "spike"
            }
        ],
        "summary": {
            "total": 1,
            "high": 1,
            "medium": 0,
            "low": 0,
            "spikes": 1,
            "drops": 0
        },
        "chartData": [],
        "avgDaily": 400
    }

