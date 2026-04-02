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
            "hourlyProfile": [],
            "heatmap": [],
            "aiRecommendation": None,
            "assets": []
        }

    import pandas as pd
    import numpy as np

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ─── BASIC STATS ─────────────────────────────
    total_consumption = df["consumption"].sum()
    avg_daily = df.groupby(df["timestamp"].dt.date)["consumption"].sum().mean()
    peak_demand = df["consumption"].max()
    baseload = df["consumption"].quantile(0.1)
    load_factor = avg_daily / peak_demand if peak_demand else 0

    # ─── HOURLY PROFILE ─────────────────────────
    df["hour"] = df["timestamp"].dt.hour

    hourly_avg = df.groupby("hour")["consumption"].mean().reset_index()

    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    weekday_avg = df[~df["is_weekend"]].groupby("hour")["consumption"].mean()
    weekend_avg = df[df["is_weekend"]].groupby("hour")["consumption"].mean()

    hourly_profile = []
    for _, row in hourly_avg.iterrows():
        hour = int(row["hour"])
        hourly_profile.append({
            "hour": f"{hour:02d}:00",
            "average": float(row["consumption"]),
            "weekday": float(weekday_avg.get(hour, 0)),
            "weekend": float(weekend_avg.get(hour, 0))
        })

    # ─── HEATMAP (7x24 simplified → frontend can reshape) ─────
    df["day"] = df["timestamp"].dt.weekday  # 0=Mon

    heatmap_df = df.groupby(["day", "hour"])["consumption"].mean().reset_index()

    max_val = heatmap_df["consumption"].max() if not heatmap_df.empty else 1

    heatmap = [
        {
            "day": int(row["day"]),
            "hour": int(row["hour"]),
            "value": float(row["consumption"]),
            "intensity": float(row["consumption"] / max_val) if max_val else 0
        }
        for _, row in heatmap_df.iterrows()
    ]

    # ─── AI RECOMMENDATION (RULE-BASED FOR NOW) ───────────────
    savings = total_consumption * 0.1  # assume 10% saving potential
    monthly_cost = total_consumption * 0.15  # £0.15 per kWh approx
    co2 = total_consumption * 0.233  # kg CO2 per kWh (UK approx)

    ai_recommendation = {
        "summary": "Shift peak loads and optimize baseload to reduce energy consumption by ~10%",
        "monthlyCost": float(monthly_cost),
        "co2Impact": float(co2),
        "potentialSavings": float(savings)
    }

    # ─── ASSETS (SIMULATED BREAKDOWN) ───────────────────────
    assets = [
        {
            "name": "HVAC",
            "share": 40,
            "cost": float(monthly_cost * 0.4),
            "carbon": float(co2 * 0.4),
            "trend": 5,
            "trendDir": "up",
            "status": "inefficient"
        },
        {
            "name": "Lighting",
            "share": 20,
            "cost": float(monthly_cost * 0.2),
            "carbon": float(co2 * 0.2),
            "trend": -2,
            "trendDir": "down",
            "status": "optimal"
        },
        {
            "name": "Equipment",
            "share": 40,
            "cost": float(monthly_cost * 0.4),
            "carbon": float(co2 * 0.4),
            "trend": 1,
            "trendDir": "up",
            "status": "moderate"
        }
    ]

    # ─── FINAL RESPONSE ─────────────────────────
    return {
        "stats": {
            "baseload": float(baseload),
            "peakDemand": float(peak_demand),
            "loadFactor": float(load_factor),
            "avgDaily": float(avg_daily)
        },
        "hourlyProfile": hourly_profile,
        "heatmap": heatmap,
        "aiRecommendation": ai_recommendation,
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
