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
def get_analytics():
    data = supabase.table("energy_data").select("*").execute().data

    if not data:
        return {
            "stats": {
                "baseload": 0,
                "peakDemand": 0,
                "loadFactor": 0,
                "avgDaily": 0
            },
            "hourlyProfile": []
        }

    import pandas as pd

    df = pd.DataFrame(data)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # ─── HOURLY PROFILE ───
    hourly = df.groupby("hour")["consption"].mean().reset_index()

    hourly_profile = [
        {
            "hour": f"{int(row['hour']):02d}:00",
            "average": round(row["consption"], 2),
            "weekday": round(row["consption"], 2),
            "weekend": round(row["consption"], 2),
        }
        for _, row in hourly.iterrows()
    ]

    # ─── STATS ───
    baseload = df["consumption"].quantile(0.1)
    peak = df["consumption"].max()
    avg = df["consumption"].mean()

    stats = {
        "baseload": round(baseload, 2),
        "peakDemand": round(peak, 2),
        "loadFactor": round(avg / peak, 2) if peak else 0,
        "avgDaily": round(avg * 24, 2),
    }

    return {
        "stats": stats,
        "hourlyProfile": hourly_profile
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
