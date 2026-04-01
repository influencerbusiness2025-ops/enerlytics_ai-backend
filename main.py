from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
import os
from supabase import create_client

app = FastAPI()

# ─── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SUPABASE INIT ─────────────────────────────────────────────
SUPABASE_URL = os.getenv("https://fopzbnloivgxzupxvhcr.supabase.co")
SUPABASE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvcHpibmxvaXZneHp1cHh2aGNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5Nzk5ODcsImV4cCI6MjA5MDU1NTk4N30.GC0Rs6N79vcXuyVBCqpyS5xH76sJ-Ea2CrY22gPyDMs")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── ROOT ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Enerlytics API running 🚀"}

# ─── UPLOAD DATA ───────────────────────────────────────────────
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    contents = await file.read()

    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        return {"success": False, "message": "Missing 'Date' column"}

    time_columns = [col for col in df.columns if ":" in col]

    df_long = df.melt(
        id_vars=["Date"],
        value_vars=time_columns,
        var_name="time",
        value_name="consumption"
    )

    df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
    df_long = df_long.dropna(subset=["consumption"])

    df_long["timestamp"] = pd.to_datetime(
        df_long["Date"] + " " + df_long["time"],
        dayfirst=True,
        errors="coerce"
    )

    df_long = df_long.dropna(subset=["timestamp"])

    df_final = df_long[["timestamp", "consumption"]]

    # 🔥 SAVE TO SUPABASE
    records = df_final.to_dict(orient="records")

    # Insert in batches (important)
    batch_size = 500
    for i in range(0, len(records), batch_size):
        supabase.table("energy_data").insert(records[i:i+batch_size]).execute()

    return {
        "success": True,
        "rowsProcessed": len(records),
        "message": "Saved to database"
    }

# ─── ANALYTICS ─────────────────────────────────────────────────
@app.get("/analytics")
def analytics():
    result = supabase.table("energy_data").select("*").execute()

    data = result.data

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

    daily = df.groupby("date")["consumption"].sum().reset_index()

    df["week"] = df["timestamp"].dt.to_period("W").astype(str)
    weekly = df.groupby("week")["consumption"].sum().reset_index()

    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["consumption"].sum().reset_index()

    total = df["consumption"].sum()

    return {
        "stats": {
            "totalConsumption": float(total),
            "avgDaily": float(daily["consumption"].mean()),
            "peakDemand": float(df["consumption"].max()),
            "peakDay": str(daily.loc[daily["consumption"].idxmax(), "date"]),
            "estimatedCost": float(total * 0.15),
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
            {"week": r["week"], "consumption": float(r["consumption"])}
            for _, r in weekly.iterrows()
        ],
        "monthly": [
            {"month": r["month"], "consumption": float(r["consumption"])}
            for _, r in monthly.iterrows()
        ]
    }
