from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import pandas as pd
from io import StringIO

# ─── SUPABASE CONFIG ───────────────────────────────────────────

SUPABASE_URL = "https://fopzbnloivgxzupxvhcr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvcHpibmxvaXZneHp1cHh2aGNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5Nzk5ODcsImV4cCI6MjA5MDU1NTk4N30.GC0Rs6N79vcXuyVBCqpyS5xH76sJ-Ea2CrY22gPyDMs"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# ─── CORS ──────────────────────────────────────────────────────

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

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
try:
contents = await file.read()

```
    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        return {"success": False, "message": "Missing 'Date' column"}

    time_columns = [col for col in df.columns if ":" in col]

    if not time_columns:
        return {"success": False, "message": "No time columns found"}

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

    records = df_final.to_dict(orient="records")

    # Insert in batches (important for large files)
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        response = supabase.table("energy_data").insert(batch).execute()
        print("Inserted batch:", i, "→", i + len(batch))

    return {
        "success": True,
        "rowsProcessed": len(records),
        "message": "Data stored in Supabase"
    }

except Exception as e:
    print("UPLOAD ERROR:", str(e))
    return {"success": False, "message": str(e)}
```

# ─── ANALYTICS ─────────────────────────────────────────────────

@app.get("/analytics")
def analytics():
try:
response = supabase.table("energy_data").select("*").execute()

```
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

    daily = df.groupby("date")["consumption"].sum().reset_index()

    df["week"] = df["timestamp"].dt.isocalendar().week
    weekly = df.groupby("week")["consumption"].sum().reset_index()

    df["month"] = df["timestamp"].dt.strftime("%b")
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
            {"date": str(row["date"]), "label": str(row["date"]), "consumption": float(row["consumption"])}
            for _, row in daily.iterrows()
        ],
        "weekly": [
            {"week": f"Week {int(row['week'])}", "consumption": float(row["consumption"])}
            for _, row in weekly.iterrows()
        ],
        "monthly": [
            {"month": row["month"], "consumption": float(row["consumption"])}
            for _, row in monthly.iterrows()
        ]
    }

except Exception as e:
    print("ANALYTICS ERROR:", str(e))
    return {"error": str(e)}
```

# ─── ANOMALIES (TEMP) ──────────────────────────────────────────

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
