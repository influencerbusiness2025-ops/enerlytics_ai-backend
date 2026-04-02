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

@app.get("/health")
def health():
    return {"status": "ok"}

# ─── UPLOAD CSV ───────────────────────────────────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Detect date column
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if not date_col:
            return {"success": False, "message": "No date column found"}

        # Detect time columns
        time_columns = [col for col in df.columns if ":" in col]
        if not time_columns:
            return {"success": False, "message": "No time columns found"}

        # Melt
        df_long = df.melt(
            id_vars=[date_col],
            value_vars=time_columns,
            var_name="time",
            value_name="consumption"
        )

        # Clean numeric
        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.dropna(subset=["consumption"])

        # Timestamp
        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"],
            dayfirst=True,
            errors="coerce"
        )

        df_long = df_long.dropna(subset=["timestamp"])

        df_final = df_long[["timestamp", "consumption"]]

        if df_final.empty:
            return {"success": False, "message": "No valid data after processing"}

        # 🔥 IMPORTANT: convert to proper types
        df_final["timestamp"] = df_final["timestamp"].astype(str)
        df_final["consumption"] = df_final["consumption"].astype(float)

        records = df_final.to_dict(orient="records")

        print("Total rows:", len(records))

        # 🔥 BATCH INSERT + ERROR CHECK
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            response = supabase.table("energy_data").insert(batch).execute()

            if hasattr(response, "error") and response.error:
                print("SUPABASE ERROR:", response.error)
                return {
                    "success": False,
                    "message": f"Supabase error: {response.error}"
                }

        return {
            "success": True,
            "rowsProcessed": len(records),
            "message": "Data stored in database"
        }

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {
            "success": False,
            "message": str(e)
        }

# ─── ANALYTICS ────────────────────────────────────────────────

@app.get("/analytics")
def analytics():
    response = supabase.table("energy_data").select("*").execute()
    data = response.data

    if not data:
        return {
            "stats": {},
            "daily": [],
            "weekly": [],
            "monthly": [],
            "hourlyProfile": [],
            "heatmap": [],
            "aiRecommendation": None,
            "assets": []
        }

    import pandas as pd
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ─── CORE ─────────────────────────────
    total = df["consumption"].sum()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["week"] = df["timestamp"].dt.isocalendar().week
    df["month"] = df["timestamp"].dt.strftime("%b")

    daily_df = df.groupby("date")["consumption"].sum().reset_index()
    weekly_df = df.groupby("week")["consumption"].sum().reset_index()
    monthly_df = df.groupby("month")["consumption"].sum().reset_index()

    avg_daily = daily_df["consumption"].mean()
    peak = df["consumption"].max()
    baseload = df["consumption"].quantile(0.1)

    # ─── TREND ───────────────────────────
    if len(daily_df) >= 14:
        last = daily_df.tail(7)["consumption"].sum()
        prev = daily_df.iloc[-14:-7]["consumption"].sum()
        trend = ((last - prev) / prev * 100) if prev else 0
    else:
        trend = 0

    # ─── HOURLY ──────────────────────────
    hourly_df = df.groupby("hour")["consumption"].mean().reset_index()

    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    weekday_avg = df[~df["is_weekend"]].groupby("hour")["consumption"].mean()
    weekend_avg = df[df["is_weekend"]].groupby("hour")["consumption"].mean()

    hourly_profile = [
        {
            "hour": f"{int(r['hour']):02d}:00",
            "average": float(r["consumption"]),
            "weekday": float(weekday_avg.get(int(r["hour"]), 0)),
            "weekend": float(weekend_avg.get(int(r["hour"]), 0))
        }
        for _, r in hourly_df.iterrows()
    ]

    # ─── HEATMAP ─────────────────────────
    df["day"] = df["timestamp"].dt.weekday
    heatmap_df = df.groupby(["day", "hour"])["consumption"].mean().reset_index()
    max_val = heatmap_df["consumption"].max()

    heatmap = [
        {
            "day": int(r["day"]),
            "hour": int(r["hour"]),
            "value": float(r["consumption"]),
            "intensity": float(r["consumption"] / max_val) if max_val else 0
        }
        for _, r in heatmap_df.iterrows()
    ]

    # ─── AI (basic for now) ───────────────
    ai = {
        "summary": "Energy peaks detected. Shifting load can reduce costs.",
        "monthlyCost": float(total * 0.15),
        "co2Impact": float(total * 0.233)
    }

    # ─── ASSETS (placeholder split) ───────
    assets = [
        {"name": "HVAC", "share": 40, "cost": total*0.4*0.15, "carbon": total*0.4*0.233, "trend": 5, "trendDir": "up", "status": "inefficient"},
        {"name": "Lighting", "share": 20, "cost": total*0.2*0.15, "carbon": total*0.2*0.233, "trend": -2, "trendDir": "down", "status": "optimal"},
        {"name": "Equipment", "share": 40, "cost": total*0.4*0.15, "carbon": total*0.4*0.233, "trend": 1, "trendDir": "up", "status": "moderate"}
    ]

    return {
        "stats": {
            "totalConsumption": float(total),
            "avgDaily": float(avg_daily),
            "peakDemand": float(peak),
            "peakDay": str(daily_df.loc[daily_df["consumption"].idxmax(), "date"]),
            "estimatedCost": float(total * 0.15),
            "baseload": float(baseload),
            "daysOfData": int(len(daily_df)),
            "trend": {
                "consumptionChange": float(trend),
                "costChange": float(trend)
            }
        },
        "daily": [
            {"date": str(r["date"]), "label": str(r["date"]), "consumption": float(r["consumption"])}
            for _, r in daily_df.iterrows()
        ],
        "weekly": [
            {"week": f"Week {int(r['week'])}", "consumption": float(r["consumption"])}
            for _, r in weekly_df.iterrows()
        ],
        "monthly": [
            {"month": r["month"], "consumption": float(r["consumption"])}
            for _, r in monthly_df.iterrows()
        ],
        "hourlyProfile": hourly_profile,
        "heatmap": heatmap,
        "aiRecommendation": ai,
        "assets": assets
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
# ─── DELETE DATA ──────────────────────────────────────────────

@app.delete("/delete-data")
def delete_data():
    try:
        response = supabase.table("energy_data").delete().gt("id", "00000000-0000-0000-0000-000000000000").execute()

        return {
            "success": True,
            "message": "All data deleted successfully"
        }

    except Exception as e:
        print("DELETE ERROR:", str(e))
        return {
            "success": False,
            "message": str(e)
        }
