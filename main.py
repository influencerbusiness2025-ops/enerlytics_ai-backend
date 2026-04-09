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

# ─── HELPERS ──────────────────────────────────────────────────

def parse_timestamps_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps stored as naive local wall-clock strings.
    NO utc=True, NO tz_convert — the strings are already in Europe/London time.
    e.g. "2024-01-16 08:30:00" stays as 08:30, not shifted to 09:30.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    print(f"  Parsed timestamps (first 3): {df['timestamp'].head(3).tolist()}")
    print(f"  Unique hours found: {sorted(df['timestamp'].dt.hour.unique())}")
    return df


def build_hourly_profile(df: pd.DataFrame) -> list:
    """Build 24-slot hourly profile with weekday/weekend split."""
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    hourly_profile = []
    for h in range(24):
        hour_df = df[df["hour"] == h]
        if hour_df.empty:
            hourly_profile.append({"hour": f"{h:02d}:00", "average": 0, "weekday": 0, "weekend": 0})
            continue

        avg = hour_df["consumption"].mean()
        weekday_df = hour_df[~hour_df["is_weekend"]]
        weekend_df = hour_df[hour_df["is_weekend"]]
        weekday_avg = weekday_df["consumption"].mean() if not weekday_df.empty else avg
        weekend_avg = weekend_df["consumption"].mean() if not weekend_df.empty else avg

        hourly_profile.append({
            "hour": f"{h:02d}:00",
            "average": round(float(avg), 2),
            "weekday": round(float(weekday_avg), 2),
            "weekend": round(float(weekend_avg), 2),
        })
    return hourly_profile


def build_stats(df: pd.DataFrame) -> dict:
    baseload = df["consumption"].quantile(0.1)
    peak = df["consumption"].max()
    avg = df["consumption"].mean()
    return {
        "baseload": round(float(baseload), 2),
        "peakDemand": round(float(peak), 2),
        "loadFactor": round(float(avg / peak), 2) if peak else 0,
        "avgDaily": round(float(avg * 24), 2),
    }


# ─── ROOT ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Enerlytics API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ─── UPLOAD CSV ───────────────────────────────────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), index_col=None)

        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if not date_col:
            return {"success": False, "message": "No date column found"}

        time_columns = [col for col in df.columns if ":" in col]
        if not time_columns:
            return {"success": False, "message": "No time columns found"}

        print(f"[upload] date_col={date_col}, time_columns count={len(time_columns)}")
        print(f"[upload] Sample time columns: {time_columns[:5]}")

        df_long = df.melt(
            id_vars=[date_col],
            value_vars=time_columns,
            var_name="time",
            value_name="consumption"
        )

        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.dropna(subset=["consumption"])

        # Combine date + time column header → naive local timestamp
        # e.g. "16/01/2024" + "08:30" → "2024-01-16 08:30:00"
        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )

        print(f"[upload] Sample timestamps: {df_long['timestamp'].head(5).tolist()}")
        print(f"[upload] Valid timestamps: {df_long['timestamp'].notna().sum()} / {len(df_long)}")

        df_long = df_long.dropna(subset=["timestamp"])

        # Aggregate: sum all meters that share the same timestamp (handles multi-meter CSVs)
        df_agg = df_long.groupby("timestamp", as_index=False)["consumption"].sum()

        print(f"[upload] Rows before aggregation: {len(df_long)}, after: {len(df_agg)}")

        df_final = df_agg.copy()

        if df_final.empty:
            return {"success": False, "message": "No valid data after processing"}

        # Store as naive ISO string — no UTC suffix, no tz info
        df_final["timestamp"] = df_final["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_final["consumption"] = df_final["consumption"].astype(float)

        records = df_final.to_dict(orient="records")
        print(f"[upload] Total rows to insert: {len(records)}")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table("energy_data").insert(batch).execute()
            if hasattr(response, "error") and response.error:
                print("SUPABASE ERROR:", response.error)
                return {"success": False, "message": f"Supabase error: {response.error}"}

        return {
            "success": True,
            "rowsProcessed": len(records),
            "message": "Data stored in database"
        }

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── ANALYTICS ────────────────────────────────────────────────

@app.get("/analytics")
def get_analytics():
    data = supabase.table("energy_data").select("*").execute().data

    if not data:
        return {
            "stats": {"baseload": 0, "peakDemand": 0, "loadFactor": 0, "avgDaily": 0},
            "hourlyProfile": [],
            "daily": [],
            "totalConsumption": 0,
            "heatmap": [[0.0] * 24 for _ in range(7)],
        }

    df = pd.DataFrame(data)
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df = df.dropna(subset=["consumption"])

    # FIX: parse as naive — no utc=True, no tz_convert
    df = parse_timestamps_naive(df)

    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["date"] = df["timestamp"].dt.date

    hourly_profile = build_hourly_profile(df)
    stats = build_stats(df)
    total_consumption = round(float(df["consumption"].sum()), 2)

    # ─── DAILY BREAKDOWN with per-day hourly array ───
    daily = df.groupby("date")["consumption"].sum().reset_index()
    daily_breakdown = []
    for _, row in daily.iterrows():
        date_val = row["date"]
        day_df = df[df["date"] == date_val]
        hourly_values = [
            round(float(day_df[day_df["hour"] == h]["consumption"].sum()), 2)
            for h in range(24)
        ]
        daily_breakdown.append({
            "date": str(date_val),
            "consumption": round(float(row["consumption"]), 2),
            "hourly": hourly_values,
        })

    # ─── HEATMAP (day of week × hour of day) ───
    heatmap = [[0.0] * 24 for _ in range(7)]
    counts  = [[0]   * 24 for _ in range(7)]
    for _, row in df.iterrows():
        d = row["timestamp"].dayofweek
        h = row["timestamp"].hour
        heatmap[d][h] += row["consumption"]
        counts[d][h]  += 1
    for d in range(7):
        for h in range(24):
            if counts[d][h] > 0:
                heatmap[d][h] = round(heatmap[d][h] / counts[d][h], 2)

    return {
        "stats": stats,
        "hourlyProfile": hourly_profile,
        "daily": daily_breakdown,
        "totalConsumption": total_consumption,
        "heatmap": heatmap,
    }

# ─── HOURLY PROFILE BY YEAR ───────────────────────────────────

@app.get("/analytics/hourly-profile/{year}")
def get_hourly_profile_by_year(year: int):
    try:
        data = (
            supabase.table("energy_data")
            .select("timestamp, consumption")
            .gte("timestamp", f"{year}-01-01")
            .lte("timestamp", f"{year}-12-31T23:59:59")
            .execute()
            .data
        )

        if not data:
            return {
                "hourlyProfile": [
                    {"hour": f"{h:02d}:00", "average": 0, "weekday": 0, "weekend": 0}
                    for h in range(24)
                ]
            }

        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])

        # FIX: parse as naive — no utc=True, no tz_convert
        df = parse_timestamps_naive(df)

        return {"hourlyProfile": build_hourly_profile(df)}

    except Exception as e:
        print("HOURLY PROFILE BY YEAR ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── UPLOAD GAS CSV ───────────────────────────────────────────

@app.post("/upload-gas-data")
async def upload_gas_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), index_col=None)

        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if not date_col:
            return {"success": False, "message": "No date column found"}

        time_columns = [col for col in df.columns if ":" in col]
        if not time_columns:
            return {"success": False, "message": "No time columns found"}

        df_long = df.melt(
            id_vars=[date_col],
            value_vars=time_columns,
            var_name="time",
            value_name="consumption"
        )

        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.dropna(subset=["consumption"])

        print(f"[gas-upload] Sample time values: {df_long['time'].head(5).tolist()}")

        # Combine date + time column header → naive local timestamp
        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )

        print(f"[gas-upload] Sample timestamps: {df_long['timestamp'].head(5).tolist()}")
        print(f"[gas-upload] Valid timestamps: {df_long['timestamp'].notna().sum()} / {len(df_long)}")

        df_long = df_long.dropna(subset=["timestamp"])

        # Aggregate: sum all meters that share the same timestamp (handles multi-meter CSVs)
        df_agg = df_long.groupby("timestamp", as_index=False)["consumption"].sum()

        print(f"[gas-upload] Rows before aggregation: {len(df_long)}, after: {len(df_agg)}")

        df_final = df_agg.copy()

        if df_final.empty:
            return {"success": False, "message": "No valid data after processing"}

        # Store as naive ISO string — no UTC suffix
        df_final["timestamp"] = df_final["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_final["consumption"] = df_final["consumption"].astype(float)

        records = df_final.to_dict(orient="records")
        print(f"[gas-upload] Total rows to insert: {len(records)}")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table("gas_data").insert(batch).execute()
            if hasattr(response, "error") and response.error:
                print("SUPABASE GAS ERROR:", response.error)
                return {"success": False, "message": f"Supabase error: {response.error}"}

        return {
            "success": True,
            "rowsProcessed": len(records),
            "message": "Gas data stored in database"
        }

    except Exception as e:
        print("GAS UPLOAD ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── GAS ANALYTICS ────────────────────────────────────────────

@app.get("/gas-analytics")
def get_gas_analytics():
    data = supabase.table("gas_data").select("*").execute().data

    if not data:
        return {
            "stats": {"baseload": 0, "peakDemand": 0, "loadFactor": 0, "avgDaily": 0},
            "hourlyProfile": [],
            "daily": [],
            "totalConsumption": 0,
        }

    df = pd.DataFrame(data)
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df = df.dropna(subset=["consumption"])

    # FIX: parse as naive — no utc=True, no tz_convert
    df = parse_timestamps_naive(df)

    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["consumption"].sum().reset_index()
    daily_breakdown = [
        {"date": str(row["date"]), "consumption": round(float(row["consumption"]), 2)}
        for _, row in daily.iterrows()
    ]

    return {
        "stats": build_stats(df),
        "hourlyProfile": build_hourly_profile(df),
        "daily": daily_breakdown,
        "totalConsumption": round(float(df["consumption"].sum()), 2),
    }

# ─── ANOMALIES ────────────────────────────────────────────────

@app.get("/anomalies")
def anomalies():
    return {
        "anomalies": [],
        "summary": {"total": 0, "high": 0, "medium": 0, "low": 0, "spikes": 0, "drops": 0},
        "chartData": [],
        "avgDaily": 0
    }

# ─── DEBUG: DATA SUMMARY ──────────────────────────────────────

@app.get("/debug/data-summary")
def debug_data_summary():
    try:
        data = supabase.table("energy_data").select("*").execute().data
        if not data:
            return {"rowCount": 0, "dateRange": {"earliest": None, "latest": None},
                    "totalConsumption": 0, "avgConsumption": 0, "minConsumption": 0,
                    "maxConsumption": 0, "peakDemand": 0, "avgDaily": 0}

        df = pd.DataFrame(data)
        df = parse_timestamps_naive(df)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df["date"] = df["timestamp"].dt.date

        # Key check: confirm hours are spread correctly
        hour_dist = df.groupby(df["timestamp"].dt.hour)["consumption"].count().to_dict()
        print(f"[debug] Hour distribution: {hour_dist}")

        unique_dates = df["date"].nunique()
        total = float(df["consumption"].sum())
        return {
            "rowCount": len(df),
            "dateRange": {"earliest": str(df["timestamp"].min()), "latest": str(df["timestamp"].max())},
            "totalConsumption": round(total, 2),
            "avgConsumption": round(float(df["consumption"].mean()), 2),
            "minConsumption": round(float(df["consumption"].min()), 2),
            "maxConsumption": round(float(df["consumption"].max()), 2),
            "peakDemand": round(float(df["consumption"].max()), 2),
            "avgDaily": round(total / unique_dates, 2) if unique_dates else 0,
            "hourDistribution": hour_dist,  # now returned in API response for easy verification
        }

    except Exception as e:
        print("DEBUG SUMMARY ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── DEBUG: GAS DATA SUMMARY ──────────────────────────────────

@app.get("/debug/gas-summary")
def debug_gas_summary():
    try:
        data = supabase.table("gas_data").select("*").execute().data
        if not data:
            return {"rowCount": 0, "dateRange": {"earliest": None, "latest": None},
                    "totalConsumption": 0, "avgConsumption": 0, "minConsumption": 0,
                    "maxConsumption": 0, "peakDemand": 0, "avgDaily": 0}

        df = pd.DataFrame(data)
        df = parse_timestamps_naive(df)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df["date"] = df["timestamp"].dt.date

        hour_dist = df.groupby(df["timestamp"].dt.hour)["consumption"].count().to_dict()
        unique_dates = df["date"].nunique()
        total = float(df["consumption"].sum())
        return {
            "rowCount": len(df),
            "dateRange": {"earliest": str(df["timestamp"].min()), "latest": str(df["timestamp"].max())},
            "totalConsumption": round(total, 2),
            "avgConsumption": round(float(df["consumption"].mean()), 2),
            "minConsumption": round(float(df["consumption"].min()), 2),
            "maxConsumption": round(float(df["consumption"].max()), 2),
            "peakDemand": round(float(df["consumption"].max()), 2),
            "avgDaily": round(total / unique_dates, 2) if unique_dates else 0,
            "hourDistribution": hour_dist,
        }

    except Exception as e:
        print("GAS DEBUG SUMMARY ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── DELETE DATA ──────────────────────────────────────────────

@app.delete("/delete-data")
def delete_data():
    try:
        supabase.table("energy_data").delete().gt("id", "00000000-0000-0000-0000-000000000000").execute()
        return {"success": True, "message": "All energy data deleted successfully"}
    except Exception as e:
        print("DELETE ERROR:", str(e))
        return {"success": False, "message": str(e)}

@app.delete("/delete-gas-data")
def delete_gas_data():
    try:
        supabase.table("gas_data").delete().gt("id", "00000000-0000-0000-0000-000000000000").execute()
        return {"success": True, "message": "All gas data deleted successfully"}
    except Exception as e:
        print("DELETE GAS ERROR:", str(e))
        return {"success": False, "message": str(e)}
