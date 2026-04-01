from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO

app = FastAPI()

# ─── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── GLOBAL STORAGE (TEMP - replace with DB later) ──────────────
stored_data = None

# ─── ROOT ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Enerlytics API running 🚀"}

# ─── HEALTH ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─── UPLOAD DATA ───────────────────────────────────────────────
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    global stored_data

    contents = await file.read()

    # Read CSV
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Detect time columns (HH:MM format)
    time_columns = [col for col in df.columns if ":" in col]

    if not time_columns:
        return {"success": False, "message": "No time-based columns found"}

    # Convert wide → long
    df_long = df.melt(
        id_vars=["Date"],
        value_vars=time_columns,
        var_name="time",
        value_name="consumption"
    )

    # Convert to numeric
    df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")

    # Drop invalid values
    df_long = df_long.dropna(subset=["consumption"])

    # Create timestamp
    df_long["timestamp"] = pd.to_datetime(
        df_long["Date"] + " " + df_long["time"],
        errors="coerce"
    )

    df_long = df_long.dropna(subset=["timestamp"])

    # Final dataset
    df_final = df_long[["timestamp", "consumption"]].sort_values("timestamp")

    # Store globally
    stored_data = df_final

    return {
        "success": True,
        "rowsProcessed": len(df_final),
        "message": "File processed successfully"
    }

# ─── ANALYTICS ─────────────────────────────────────────────────
@app.get("/analytics")
def analytics():
    global stored_data

    if stored_data is None:
        return {
            "stats": {
                "totalConsumption": 0,
                "avgDaily": 0,
                "peakDemand": 0,
                "peakDay": "N/A",
                "estimatedCost": 0,
                "baseload": 0,
                "daysOfData": 0,
                "trend": {
                    "consumptionChange": 0,
                    "costChange": 0
                }
            },
            "daily": [],
            "weekly": [],
            "monthly": []
        }

    df = stored_data.copy()

    # Create date column
    df["date"] = df["timestamp"].dt.date

    # DAILY
    daily = df.groupby("date")["consumption"].sum().reset_index()

    # WEEKLY
    df["week"] = df["timestamp"].dt.isocalendar().week
    weekly = df.groupby("week")["consumption"].sum().reset_index()

    # MONTHLY
    df["month"] = df["timestamp"].dt.strftime("%b")
    monthly = df.groupby("month")["consumption"].sum().reset_index()

    # STATS
    total_consumption = df["consumption"].sum()
    avg_daily = daily["consumption"].mean()
    peak_demand = df["consumption"].max()

    return {
        "stats": {
            "totalConsumption": float(total_consumption),
            "avgDaily": float(avg_daily),
            "peakDemand": float(peak_demand),
            "peakDay": str(daily.loc[daily["consumption"].idxmax(), "date"]),
            "estimatedCost": float(total_consumption * 0.15),
            "baseload": float(df["consumption"].quantile(0.1)),
            "daysOfData": int(len(daily)),
            "trend": {
                "consumptionChange": 0,
                "costChange": 0
            }
        },
        "daily": [
            {
                "date": str(row["date"]),
                "label": str(row["date"]),
                "consumption": float(row["consumption"])
            }
            for _, row in daily.iterrows()
        ],
        "weekly": [
            {
                "week": f"Week {int(row['week'])}",
                "consumption": float(row["consumption"])
            }
            for _, row in weekly.iterrows()
        ],
        "monthly": [
            {
                "month": row["month"],
                "consumption": float(row["consumption"])
            }
            for _, row in monthly.iterrows()
        ]
    }

# ─── ANOMALIES (TEMP MOCK) ─────────────────────────────────────
@app.get("/anomalies")
def get_anomalies():
    return {
        "anomalies": [],
        "summary": {
            "total": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "spikes": 0,
            "drops": 0
        },
        "chartData": [],
        "avgDaily": 0
    }
