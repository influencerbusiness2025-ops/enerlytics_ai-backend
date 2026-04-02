from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
from supabase import create_client

# ─── SUPABASE CONFIG ───────────────────────────────────────────

SUPABASE_URL = "https://fopzbnloivgxzupxvhcr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvcHpibmxvaXZneHp1cHh2aGNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5Nzk5ODcsImV4cCI6MjA5MDU1NTk4N30.GC0Rs6N79vcXuyVBCqpyS5xH76sJ-Ea2CrY22gPyDMs"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── APP INIT ─────────────────────────────────────────────────

app = FastAPI()

# ─── CORS ─────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── ROOT ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Enerlytics API running 🚀"}

# ─── HEALTH ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

# ─── UPLOAD CSV ───────────────────────────────────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()

    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Detect date column automatically
    date_col = next((c for c in df.columns if "date" in c.lower()), None)

    if not date_col:
        return {"success": False, "message": "No date column found"}

    # Detect time columns
    time_columns = [col for col in df.columns if ":" in col]

    if not time_columns:
        return {"success": False, "message": "No time columns found"}

    # Convert wide → long
    df_long = df.melt(
        id_vars=[date_col],
        value_vars=time_columns,
        var_name="time",
        value_name="consumption"
    )

    # Clean data
    df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
    df_long = df_long.dropna(subset=["consumption"])

    # Create timestamp
    df_long["timestamp"] = pd.to_datetime(
        df_long[date_col].astype(str) + " " + df_long["time"],
        errors="coerce"
    )

    df_long = df_long.dropna(subset=["timestamp"])

    df_final = df_long[["timestamp", "consumption"]]

    # Convert to list
    records = df_final.to_dict(orient="records")

    if not records:
        return {"success": False, "message": "No valid data found"}

    # Insert into Supabase (batch safe)
    supabase.table("energy_data").insert(records).execute()

    return {
        "success": True,
        "rowsProcessed": len(records),
        "message": "Data stored in database"
    }

# ─── ANALYTICS ────────────────────────────────────────────────

@app.get("/analytics")
def analytics():
    response = supabase.table("energy_data").select("*").execute()
    data = response.data

    if not data:
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

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["date"] = df["timestamp"].dt.date
    df["week"] = df["timestamp"].dt.isocalendar().week
    df["month"] = df["timestamp"].dt.strftime("%b")

    # Aggregations
    daily = df.groupby("date")["consumption"].sum().reset_index()
    weekly = df.groupby("week")["consumption"].sum().reset_index()
    monthly = df.groupby("month")["consumption"].sum().reset_index()

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
            {"date": str(r["date"]), "label": str(r["date"]), "consumption": float(r["consumption"])}
            for _, r in daily.iterrows()
        ],
        "weekly": [
            {"week": f"Week {int(r['week'])}", "consumption": float(r["consumption"])}
            for _, r in weekly.iterrows()
        ],
        "monthly": [
            {"month": r["month"], "consumption": float(r["consumption"])}
            for _, r in monthly.iterrows()
        ]
    }

# ─── ANOMALIES ────────────────────────────────────────────────

@app.get("/anomalies")
def anomalies():
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
