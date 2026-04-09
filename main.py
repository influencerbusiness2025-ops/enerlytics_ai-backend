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
        df = pd.read_csv(StringIO(contents.decode("utf-8")), index_col=None)

        # Detect date column
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if not date_col:
            return {"success": False, "message": "No date column found"}

        # Detect time columns
        time_columns = [col for col in df.columns if ":" in col]
        if not time_columns:
            return {"success": False, "message": "No time columns found"}

        # ─── DEBUG: pre-melt diagnostics ───
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"Detected date column: {date_col}")
        print(f"Detected time columns: {time_columns}")
        print(f"Number of time columns: {len(time_columns)}")
        print(f"DataFrame before melt (first 3 rows):\n{df.head(3)}")

        # Melt
        df_long = df.melt(
            id_vars=[date_col],
            value_vars=time_columns,
            var_name="time",
            value_name="consumption"
        )

        # ─── DEBUG: post-melt diagnostics ───
        print(f"Melted dataframe shape: {df_long.shape}")
        print(f"Melted dataframe columns: {df_long.columns.tolist()}")
        print(f"First 30 rows of melted data:")
        print(df_long.head(30).to_string())
        print(f"Unique time values (count): {df_long['time'].nunique()}")
        print(f"Time value counts:\n{df_long['time'].value_counts().head(30)}")
        print(f"All unique time values in melted dataframe:")
        print(sorted(df_long['time'].unique().tolist()))


        # Clean numeric
        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.dropna(subset=["consumption"])

        # Debug: sample time values before padding
        print("Sample time values before padding:", df_long["time"].head(10).tolist())

        # Pad single-digit hours to HH:MM format (e.g. "0:00" → "00:00", "1:00" → "01:00")
        df_long["time"] = df_long["time"].apply(
            lambda x: f"{int(x.split(':')[0]):02d}:{x.split(':')[1]}"
        )

        # Debug: sample time values after padding
        print("Sample time values after padding:", df_long["time"].head(10).tolist())

        # Timestamp
        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"],
            dayfirst=True,
            errors="coerce"
        )

        # Debug: sample timestamps created
        print("Sample timestamps created:", df_long["timestamp"].head(10).tolist())
        print("Rows with valid timestamps:", df_long["timestamp"].notna().sum())

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


    # ─── DEBUG: raw data from Supabase ───
    print(f"Raw data from Supabase (first 3 rows): {data[:3]}")

    df = pd.DataFrame(data)

    # ─── DEBUG: DataFrame shape and types ───
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame dtypes:\n{df.dtypes}")
    print(f"First 3 rows:\n{df.head(3)}")

    # ─── DEBUG: raw timestamps from DB ───
    print(f"Sample raw timestamps from DB (first 5):\n{df['timestamp'].head().tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ─── DEBUG: after UTC parsing ───
    print(f"After UTC parsing (first 3):\n{df['timestamp'].head(3).tolist()}")

    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df = df.dropna(subset=["consumption"])

    # ─── TIMEZONE: convert UTC → Europe/London ───
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/London")

    # ─── DEBUG: after Europe/London conversion ───
    print(f"After Europe/London conversion (first 3):\n{df['timestamp'].head(3).tolist()}")

    df["hour"] = df["timestamp"].dt.hour
    # dayofweek: 0=Monday … 4=Friday → weekday; 5=Saturday, 6=Sunday → weekend
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    # ─── DEBUG: hour extraction ───
    print(f"Hour column (first 10):\n{df['hour'].head(10).tolist()}")
    print(f"Unique hours: {sorted(df['hour'].unique())}")
    print(f"Records per hour:\n{df.groupby('hour').size()}")

    # ─── HOURLY PROFILE ───
    hourly_profile = []
    for h in range(24):
        hour_df = df[df["hour"] == h]

        if hour_df.empty:
            hourly_profile.append(
                {"hour": f"{h:02d}:00", "average": 0, "weekday": 0, "weekend": 0}
            )
            continue

        avg = hour_df["consumption"].mean()

        weekday_df = hour_df[~hour_df["is_weekend"]]
        weekend_df = hour_df[hour_df["is_weekend"]]

        weekday_avg = weekday_df["consumption"].mean() if not weekday_df.empty else avg
        weekend_avg = weekend_df["consumption"].mean() if not weekend_df.empty else avg

        hourly_profile.append(
            {
                "hour": f"{h:02d}:00",
                "average": round(float(avg), 2),
                "weekday": round(float(weekday_avg), 2),
                "weekend": round(float(weekend_avg), 2),
            }
        )

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

    # ─── TOTAL CONSUMPTION ───
    total_consumption = round(float(df["consumption"].sum()), 2)

    # ─── DAILY BREAKDOWN ───
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["consumption"].sum().reset_index()

    # ─── DEBUG: per-day hourly aggregation ───
    first_date = df["date"].iloc[0]
    first_day_df = df[df["date"] == first_date]
    print(f"First date: {first_date}")
    print(f"Records for first date: {len(first_day_df)}")
    print(f"Hours in first date: {sorted(first_day_df['hour'].unique())}")
    for h in range(24):
        hour_total = first_day_df[first_day_df["hour"] == h]["consumption"].sum()
        if hour_total > 0:
            print(f"  Hour {h}: {hour_total}")

    daily_breakdown = []
    for _, row in daily.iterrows():
        date_val = row["date"]
        day_df = df[df["date"] == date_val]

        # Build 24-element hourly array for this specific day
        hourly_values = []
        for h in range(24):
            hour_total = day_df[day_df["hour"] == h]["consumption"].sum()
            hourly_values.append(round(float(hour_total), 2))

        daily_breakdown.append(
            {
                "date": str(date_val),
                "consumption": round(float(row["consumption"]), 2),
                "hourly": hourly_values,
            }
        )

    # ─── HEATMAP (day of week × hour) ───
    heatmap = [[0.0 for _ in range(24)] for _ in range(7)]
    counts = [[0 for _ in range(24)] for _ in range(7)]

    for _, row in df.iterrows():
        day = row["timestamp"].dayofweek  # 0=Monday … 6=Sunday
        hour = row["timestamp"].hour      # 0-23
        heatmap[day][hour] += row["consumption"]
        counts[day][hour] += 1

    for day in range(7):
        for hour in range(24):
            if counts[day][hour] > 0:
                heatmap[day][hour] = round(heatmap[day][hour] / counts[day][hour], 2)

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
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        data = (
            supabase.table("energy_data")
            .select("timestamp, consumption")
            .gte("timestamp", start_date)
            .lte("timestamp", end_date)
            .execute()
            .data
        )

        if not data:
            return {
                "hourlyProfile": [
                    {
                        "hour": f"{h:02d}:00",
                        "average": 0,
                        "weekday": 0,
                        "weekend": 0,
                    }
                    for h in range(24)
                ]
            }

        df = pd.DataFrame(data)

        # ─── DEBUG: raw timestamps from DB ───
        print(f"Sample raw timestamps from DB (first 5):\n{df['timestamp'].head().tolist()}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        print(f"Timestamps after UTC parsing (first 5):\n{df['timestamp'].head().tolist()}")

        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])

        # ─── TIMEZONE: convert UTC → Europe/London ───
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/London")
        print(f"Timestamps after Europe/London conversion (first 5):\n{df['timestamp'].head().tolist()}")

        df["hour"] = df["timestamp"].dt.hour
        # dayofweek: 0=Monday … 4=Friday → weekday; 5=Saturday, 6=Sunday → weekend
        df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

        print(f"Hours in data: {sorted(df['hour'].unique())}")

        hourly_profile = []
        for h in range(24):
            hour_df = df[df["hour"] == h]

            if hour_df.empty:
                hourly_profile.append(
                    {"hour": f"{h:02d}:00", "average": 0, "weekday": 0, "weekend": 0}
                )
                continue

            avg = hour_df["consumption"].mean()

            weekday_df = hour_df[~hour_df["is_weekend"]]
            weekend_df = hour_df[hour_df["is_weekend"]]

            weekday_avg = weekday_df["consumption"].mean() if not weekday_df.empty else avg
            weekend_avg = weekend_df["consumption"].mean() if not weekend_df.empty else avg

            hourly_profile.append(
                {
                    "hour": f"{h:02d}:00",
                    "average": round(float(avg), 2),
                    "weekday": round(float(weekday_avg), 2),
                    "weekend": round(float(weekend_avg), 2),
                }
            )

        return {"hourlyProfile": hourly_profile}

    except Exception as e:
        print("HOURLY PROFILE BY YEAR ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── UPLOAD GAS CSV ───────────────────────────────────────────

@app.post("/upload-gas-data")
async def upload_gas_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), index_col=None)

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

        print(f"All unique time values in melted dataframe:")
        print(sorted(df_long['time'].unique().tolist()))

        # Debug: sample time values before padding
        print("[gas] Sample time values before padding:", df_long["time"].head(10).tolist())

        # Pad single-digit hours to HH:MM format (e.g. "0:00" → "00:00", "1:00" → "01:00")
        df_long["time"] = df_long["time"].apply(
            lambda x: f"{int(x.split(':')[0]):02d}:{x.split(':')[1]}"
        )

        # Debug: sample time values after padding
        print("[gas] Sample time values after padding:", df_long["time"].head(10).tolist())

        # Timestamp
        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"],
            dayfirst=True,
            errors="coerce"
        )

        # Debug: sample timestamps created
        print("[gas] Sample timestamps created:", df_long["timestamp"].head(10).tolist())
        print("[gas] Rows with valid timestamps:", df_long["timestamp"].notna().sum())

        df_long = df_long.dropna(subset=["timestamp"])

        df_final = df_long[["timestamp", "consumption"]]

        if df_final.empty:
            return {"success": False, "message": "No valid data after processing"}

        df_final["timestamp"] = df_final["timestamp"].astype(str)
        df_final["consumption"] = df_final["consumption"].astype(float)

        records = df_final.to_dict(orient="records")

        print("Total gas rows:", len(records))

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            response = supabase.table("gas_data").insert(batch).execute()

            if hasattr(response, "error") and response.error:
                print("SUPABASE GAS ERROR:", response.error)
                return {
                    "success": False,
                    "message": f"Supabase error: {response.error}"
                }

        return {
            "success": True,
            "rowsProcessed": len(records),
            "message": "Gas data stored in database"
        }

    except Exception as e:
        print("GAS UPLOAD ERROR:", str(e))
        return {
            "success": False,
            "message": str(e)
        }

# ─── GAS ANALYTICS ────────────────────────────────────────────

@app.get("/gas-analytics")
def get_gas_analytics():
    data = supabase.table("gas_data").select("*").execute().data

    if not data:
        return {
            "stats": {
                "baseload": 0,
                "peakDemand": 0,
                "loadFactor": 0,
                "avgDaily": 0
            },
            "hourlyProfile": [],
            "daily": [],
            "totalConsumption": 0,
        }


    df = pd.DataFrame(data)

    # ─── DEBUG: raw timestamps from DB ───
    print(f"[gas] Sample raw timestamps from DB (first 5):\n{df['timestamp'].head().tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"[gas] Timestamps after UTC parsing (first 5):\n{df['timestamp'].head().tolist()}")

    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df = df.dropna(subset=["consumption"])

    # ─── TIMEZONE: convert UTC → Europe/London ───
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/London")
    print(f"[gas] Timestamps after Europe/London conversion (first 5):\n{df['timestamp'].head().tolist()}")

    df["hour"] = df["timestamp"].dt.hour
    # dayofweek: 0=Monday … 4=Friday → weekday; 5=Saturday, 6=Sunday → weekend
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    # ─── DEBUG LOGGING ───
    print(f"[gas] Hours in data: {sorted(df['hour'].unique())}")
    print(f"[gas] Records per hour:\n{df.groupby('hour').size()}")

    hourly_profile = []
    for h in range(24):
        hour_df = df[df["hour"] == h]

        if hour_df.empty:
            hourly_profile.append(
                {"hour": f"{h:02d}:00", "average": 0, "weekday": 0, "weekend": 0}
            )
            continue

        avg = hour_df["consumption"].mean()

        weekday_df = hour_df[~hour_df["is_weekend"]]
        weekend_df = hour_df[hour_df["is_weekend"]]

        weekday_avg = weekday_df["consumption"].mean() if not weekday_df.empty else avg
        weekend_avg = weekend_df["consumption"].mean() if not weekend_df.empty else avg

        hourly_profile.append(
            {
                "hour": f"{h:02d}:00",
                "average": round(float(avg), 2),
                "weekday": round(float(weekday_avg), 2),
                "weekend": round(float(weekend_avg), 2),
            }
        )

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

    # ─── TOTAL CONSUMPTION ───
    total_consumption = round(float(df["consumption"].sum()), 2)

    # ─── DAILY BREAKDOWN ───
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["consumption"].sum().reset_index()

    daily_breakdown = [
        {
            "date": str(row["date"]),
            "consumption": round(float(row["consumption"]), 2),
        }
        for _, row in daily.iterrows()
    ]

    return {
        "stats": stats,
        "hourlyProfile": hourly_profile,
        "daily": daily_breakdown,
        "totalConsumption": total_consumption,
    }

# ─── DEBUG: GAS DATA SUMMARY ──────────────────────────────────

@app.get("/debug/gas-summary")
def debug_gas_summary():
    try:
        data = supabase.table("gas_data").select("*").execute().data

        if not data:
            return {
                "rowCount": 0,
                "dateRange": {"earliest": None, "latest": None},
                "totalConsumption": 0,
                "avgConsumption": 0,
                "minConsumption": 0,
                "maxConsumption": 0,
                "peakDemand": 0,
                "avgDaily": 0,
            }

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])

        df["date"] = df["timestamp"].dt.date
        unique_dates = df["date"].nunique()

        total = float(df["consumption"].sum())
        avg_daily = round(total / unique_dates, 2) if unique_dates else 0

        return {
            "rowCount": len(df),
            "dateRange": {
                "earliest": str(df["timestamp"].min()),
                "latest": str(df["timestamp"].max()),
            },
            "totalConsumption": round(total, 2),
            "avgConsumption": round(float(df["consumption"].mean()), 2),
            "minConsumption": round(float(df["consumption"].min()), 2),
            "maxConsumption": round(float(df["consumption"].max()), 2),
            "peakDemand": round(float(df["consumption"].max()), 2),
            "avgDaily": avg_daily,
        }

    except Exception as e:
        print("GAS DEBUG SUMMARY ERROR:", str(e))
        return {"success": False, "message": str(e)}

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
# ─── DEBUG: DATA SUMMARY ──────────────────────────────────────

@app.get("/debug/data-summary")
def debug_data_summary():
    try:
        data = supabase.table("energy_data").select("*").execute().data

        if not data:
            return {
                "rowCount": 0,
                "dateRange": {"earliest": None, "latest": None},
                "totalConsumption": 0,
                "avgConsumption": 0,
                "minConsumption": 0,
                "maxConsumption": 0,
                "peakDemand": 0,
                "avgDaily": 0,
            }

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])

        df["date"] = df["timestamp"].dt.date
        unique_dates = df["date"].nunique()

        total = float(df["consumption"].sum())
        avg_daily = round(total / unique_dates, 2) if unique_dates else 0

        return {
            "rowCount": len(df),
            "dateRange": {
                "earliest": str(df["timestamp"].min()),
                "latest": str(df["timestamp"].max()),
            },
            "totalConsumption": round(total, 2),
            "avgConsumption": round(float(df["consumption"].mean()), 2),
            "minConsumption": round(float(df["consumption"].min()), 2),
            "maxConsumption": round(float(df["consumption"].max()), 2),
            "peakDemand": round(float(df["consumption"].max()), 2),
            "avgDaily": avg_daily,
        }

    except Exception as e:
        print("DEBUG SUMMARY ERROR:", str(e))
        return {"success": False, "message": str(e)}

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
