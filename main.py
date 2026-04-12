from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from io import StringIO
from supabase import create_client
import httpx
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# ─── SUPABASE CONFIG ──────────────────────────────────────────

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

# ─── PYDANTIC MODELS ──────────────────────────────────────────

class SiteCreate(BaseModel):
    name: str
    lat: float
    lng: float
    timezone: str = "UTC"
    base_temp: float = 15.5
    mode: str = "auto"
    address: str = ""
    building_type: str = "commercial"

# ─── HELPERS ──────────────────────────────────────────────────

def parse_timestamps_naive(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    print(f"  Parsed timestamps (first 3): {df['timestamp'].head(3).tolist()}")
    print(f"  Unique hours found: {sorted(df['timestamp'].dt.hour.unique())}")
    return df


def build_hourly_profile(df: pd.DataFrame) -> list:
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


def auto_base_temp(lat: float, current_base: float) -> float:
    if current_base == 15.5 and abs(lat) < 23.5:
        return 24.0 if abs(lat) < 15 else 18.0
    return current_base


def resolve_mode(mode: str, lat: float, base_temp: float) -> str:
    if mode != "auto":
        return mode
    if base_temp >= 22 or abs(lat) < 15:
        return "cdd_only"
    if base_temp >= 18 or abs(lat) < 23.5:
        return "cdd_only"
    return "both"


async def fetch_degree_days(lat: float, lng: float, base_temp: float,
                             start_date: str, end_date: str) -> dict:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lng}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean&timezone=UTC"
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    result = {}
    for date, temp in zip(data["daily"]["time"], data["daily"]["temperature_2m_mean"]):
        if temp is None:
            continue
        result[date] = {
            "mean_temp": round(temp, 1),
            "hdd": round(max(0.0, base_temp - temp), 2),
            "cdd": round(max(0.0, temp - base_temp), 2),
        }
    return result


def estimate_sensitivity(consumption: list, hdd: list, cdd: list, mode: str) -> tuple:
    n = len(consumption)
    if n < 7:
        return 0.0, 0.0, float(np.mean(consumption)) if consumption else 0.0
    y = np.array(consumption, dtype=float)
    if mode == "cdd_only":
        X = np.column_stack([np.ones(n), np.array(cdd)])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return 0.0, max(0.0, float(c[1])), float(c[0])
    elif mode == "hdd_only":
        X = np.column_stack([np.ones(n), np.array(hdd)])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return max(0.0, float(c[1])), 0.0, float(c[0])
    else:
        X = np.column_stack([np.ones(n), np.array(hdd), np.array(cdd)])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return max(0.0, float(c[1])), max(0.0, float(c[2])), float(c[0])

# ─── ROOT ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Enerlytics API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ─── SITES CRUD ───────────────────────────────────────────────

@app.get("/analytics/sites")
def get_sites():
    try:
        result = (
            supabase.table("sites")
            .select("id, name, lat, lng, timezone, base_temp, mode, address, building_type")
            .eq("is_active", True)
            .order("name")
            .execute()
        )
        return {"sites": result.data or []}
    except Exception as e:
        print("GET SITES ERROR:", str(e))
        return {"success": False, "message": str(e)}


@app.post("/analytics/sites")
def create_site(site: SiteCreate):
    try:
        base_temp = auto_base_temp(site.lat, site.base_temp)
        payload = {
            "name":          site.name,
            "lat":           site.lat,
            "lng":           site.lng,
            "timezone":      site.timezone,
            "base_temp":     base_temp,
            "mode":          site.mode,
            "address":       site.address,
            "building_type": site.building_type,
            "is_active":     True,
        }
        result = supabase.table("sites").insert(payload).execute()
        return {"success": True, "site": result.data[0] if result.data else None}
    except Exception as e:
        print("CREATE SITE ERROR:", str(e))
        return {"success": False, "message": str(e)}


@app.put("/analytics/sites/{site_id}")
def update_site(site_id: str, site: SiteCreate):
    try:
        payload = {
            "name":          site.name,
            "lat":           site.lat,
            "lng":           site.lng,
            "timezone":      site.timezone,
            "base_temp":     site.base_temp,
            "mode":          site.mode,
            "address":       site.address,
            "building_type": site.building_type,
        }
        result = (
            supabase.table("sites")
            .update(payload)
            .eq("id", site_id)
            .execute()
        )
        return {"success": True, "site": result.data[0] if result.data else None}
    except Exception as e:
        print("UPDATE SITE ERROR:", str(e))
        return {"success": False, "message": str(e)}


@app.delete("/analytics/sites/{site_id}")
def delete_site(site_id: str):
    try:
        supabase.table("sites").update({"is_active": False}).eq("id", site_id).execute()
        return {"success": True, "deleted": site_id}
    except Exception as e:
        print("DELETE SITE ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── WEATHER NORMALISED ───────────────────────────────────────

@app.get("/analytics/weather-normalised")
async def get_weather_normalised(
    site_id: str = Query(..., description="UUID from /analytics/sites"),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
):
    try:
        site_result = (
            supabase.table("sites")
            .select("*")
            .eq("id", site_id)
            .eq("is_active", True)
            .single()
            .execute()
        )
        if not site_result.data:
            return {"success": False, "message": f"Site '{site_id}' not found"}

        site = site_result.data
        base_temp = site["base_temp"]
        mode = resolve_mode(site["mode"], site["lat"], base_temp)

        today = datetime.utcnow().date()
        if not end_date:
            end_date = str(today - timedelta(days=1))
        if not start_date:
            start_date = str(today - timedelta(days=31))

        data = (
            supabase.table("energy_data")
            .select("timestamp, consumption")
            .gte("timestamp", start_date)
            .lte("timestamp", end_date + "T23:59:59")
            .range(0, 20000)
            .execute()
            .data
        )
        if not data:
            return {"success": False, "message": "No energy data for date range"}

        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = df["timestamp"].dt.date.astype(str)

        daily_df = df.groupby("date")["consumption"].sum().reset_index()
        daily_df.columns = ["date", "actual"]
        daily_df = daily_df.sort_values("date").reset_index(drop=True)

        if daily_df.empty:
            return {"success": False, "message": "No daily data after aggregation"}

        actual_start = daily_df["date"].min()
        actual_end   = daily_df["date"].max()

        degree_days = await fetch_degree_days(
            lat=site["lat"], lng=site["lng"],
            base_temp=base_temp,
            start_date=actual_start, end_date=actual_end,
        )

        rows = []
        for _, row in daily_df.iterrows():
            date = row["date"]
            dd = degree_days.get(date, {"hdd": 0.0, "cdd": 0.0, "mean_temp": None})
            rows.append({
                "date":      date,
                "actual":    float(row["actual"]),
                "hdd":       dd["hdd"],
                "cdd":       dd["cdd"],
                "mean_temp": dd["mean_temp"],
            })

        beta_h, beta_c, baseload = estimate_sensitivity(
            [r["actual"] for r in rows],
            [r["hdd"]    for r in rows],
            [r["cdd"]    for r in rows],
            mode,
        )
        print(f"[weather-norm] site={site['name']} β_h={beta_h:.2f} β_c={beta_c:.2f} baseload={baseload:.1f}")

        results = []
        for r in rows:
            weather_impact = round((r["hdd"] * beta_h) + (r["cdd"] * beta_c), 1)
            results.append({
                "date":          r["date"],
                "actual":        round(r["actual"], 1),
                "normalised":    round(r["actual"] - weather_impact, 1),
                "weatherImpact": weather_impact,
                "hdd":           r["hdd"],
                "cdd":           r["cdd"],
                "meanTemp":      r["mean_temp"],
            })

        total_actual     = round(sum(r["actual"]        for r in results), 1)
        total_normalised = round(sum(r["normalised"]    for r in results), 1)
        total_weather    = round(sum(r["weatherImpact"] for r in results), 1)

        return {
            "site": {
                "id":       site["id"],
                "name":     site["name"],
                "baseTemp": base_temp,
                "mode":     mode,
                "lat":      site["lat"],
                "lng":      site["lng"],
            },
            "dateRange": {"start": actual_start, "end": actual_end},
            "coefficients": {
                "beta_h":   round(beta_h, 3),
                "beta_c":   round(beta_c, 3),
                "baseload": round(baseload, 1),
            },
            "summary": {
                "totalActual":        total_actual,
                "totalNormalised":    total_normalised,
                "totalWeatherImpact": total_weather,
                "weatherImpactPct":   round(total_weather / total_actual * 100, 1) if total_actual else 0,
            },
            "daily": results,
        }

    except httpx.RequestError as e:
        print(f"[weather-norm] Open-Meteo fetch failed: {e}")
        return {"success": False, "message": f"Weather API error: {str(e)}"}
    except Exception as e:
        print(f"[weather-norm] ERROR: {e}")
        return {"success": False, "message": str(e)}

# ─── ANOMALIES ────────────────────────────────────────────────

@app.get("/anomalies")
def get_anomalies(
    days: int = Query(default=90, ge=7, le=365, description="Days of history to scan"),
    severity: Optional[str] = Query(default=None, description="Filter: high, medium, low"),
    anomaly_type: Optional[str] = Query(default=None, description="Filter: spike, drop"),
):
    try:
        # ── 1. Load data ──────────────────────────────────────
        end_dt   = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)

        data = (
            supabase.table("energy_data")
            .select("timestamp, consumption")
            .gte("timestamp", start_dt.strftime("%Y-%m-%dT%H:%M:%S"))
            .lte("timestamp", end_dt.strftime("%Y-%m-%dT%H:%M:%S"))
            .range(0, 20000)
            .execute()
            .data
        )

        if not data:
            return {
                "anomalies": [],
                "summary": {"total": 0, "high": 0, "medium": 0, "low": 0, "spikes": 0, "drops": 0},
                "chartData": [],
                "avgDaily": 0,
                "heatmap": [[0] * 24 for _ in range(7)],
                "totalScanned": 0,
            }

        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["date"]         = df["timestamp"].dt.date.astype(str)
        df["hour"]         = df["timestamp"].dt.hour
        df["dow"]          = df["timestamp"].dt.dayofweek       # 0=Mon, 6=Sun
        df["hour_of_week"] = df["dow"] * 24 + df["hour"]       # 0–167

        # ── 2. Baseline: median per hour-of-week slot ─────────
        # Median is more robust than mean — won't be skewed by the anomalies themselves
        baseline = (
            df.groupby("hour_of_week")["consumption"]
            .median()
            .to_dict()
        )
        df["expected"] = df["hour_of_week"].map(baseline)

        # ── 3. Std deviation per slot ─────────────────────────
        std_map = (
            df.groupby("hour_of_week")["consumption"]
            .std()
            .fillna(0)
            .to_dict()
        )
        df["std"] = df["hour_of_week"].map(std_map)

        # ── 4. Detect anomalies — tiered thresholds ───────────
        # Works for regular datasets (hotels, offices) where consumption is stable
        anomalies = []

        for _, row in df.iterrows():
            expected = row["expected"]
            actual   = row["consumption"]
            std      = row["std"]

            if expected == 0:
                continue

            deviation_pct = ((actual - expected) / expected) * 100
            std_deviation = (actual - expected) / std if std > 0 else 0

            abs_std = abs(std_deviation)
            abs_pct = abs(deviation_pct)

            # Tier 1: strong statistical signal
            if abs_std >= 3.0 and abs_pct >= 20:
                pass
            # Tier 2: moderate statistical + meaningful percentage
            elif abs_std >= 2.0 and abs_pct >= 30:
                pass
            # Tier 3: large percentage swing in otherwise quiet slot
            elif abs_std >= 1.5 and abs_pct >= 50:
                pass
            else:
                continue

            a_type = "spike" if actual > expected else "drop"

            if abs_std >= 3.0 or abs_pct >= 100:
                sev = "high"
            elif abs_std >= 2.0 or abs_pct >= 50:
                sev = "medium"
            else:
                sev = "low"

            anomalies.append({
                "timestamp":     row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "date":          row["date"],
                "hour":          int(row["hour"]),
                "hourLabel":     f"{int(row['hour']):02d}:00",
                "actual":        round(float(actual), 2),
                "expected":      round(float(expected), 2),
                "deviationPct":  round(float(deviation_pct), 1),
                "stdDeviations": round(float(std_deviation), 2),
                "severity":      sev,
                "type":          a_type,
                "dow":           int(row["dow"]),
                "dowLabel":      ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][int(row["dow"])],
            })

        # ── 5. Apply filters ──────────────────────────────────
        filtered = list(anomalies)
        if severity:
            filtered = [a for a in filtered if a["severity"] == severity.lower()]
        if anomaly_type:
            filtered = [a for a in filtered if a["type"] == anomaly_type.lower()]

        sev_order = {"high": 0, "medium": 1, "low": 2}
        filtered.sort(key=lambda x: (sev_order[x["severity"]], x["timestamp"]))

        # ── 6. Summary counts (always from full unfiltered list) ─
        summary = {
            "total":  len(anomalies),
            "high":   sum(1 for a in anomalies if a["severity"] == "high"),
            "medium": sum(1 for a in anomalies if a["severity"] == "medium"),
            "low":    sum(1 for a in anomalies if a["severity"] == "low"),
            "spikes": sum(1 for a in anomalies if a["type"] == "spike"),
            "drops":  sum(1 for a in anomalies if a["type"] == "drop"),
        }

        # ── 7. Chart data: daily totals + anomaly markers ─────
        daily = df.groupby("date")["consumption"].sum().reset_index()

        anomaly_dates = {}
        for a in anomalies:
            d = a["date"]
            if d not in anomaly_dates or sev_order[a["severity"]] < sev_order[anomaly_dates[d]["sev"]]:
                anomaly_dates[d] = {"sev": a["severity"]}

        chart_data = []
        for _, row in daily.iterrows():
            d = str(row["date"])
            chart_data.append({
                "date":         d,
                "consumption":  round(float(row["consumption"]), 2),
                "hasAnomaly":   d in anomaly_dates,
                "anomalySev":   anomaly_dates[d]["sev"] if d in anomaly_dates else None,
                "anomalyCount": sum(1 for a in anomalies if a["date"] == d),
            })

        # ── 8. Anomaly heatmap: dow × hour ────────────────────
        heatmap = [[0] * 24 for _ in range(7)]
        for a in anomalies:
            heatmap[a["dow"]][a["hour"]] += 1

        # ── 9. Average daily consumption ──────────────────────
        avg_daily = round(float(df.groupby("date")["consumption"].sum().mean()), 2)

        print(f"[anomalies] Scanned {len(df)} rows → {len(anomalies)} anomalies "
              f"(high={summary['high']}, medium={summary['medium']}, low={summary['low']})")

        return {
            "anomalies":    filtered,
            "summary":      summary,
            "chartData":    chart_data,
            "avgDaily":     avg_daily,
            "heatmap":      heatmap,
            "totalScanned": len(df),
        }

    except Exception as e:
        print("ANOMALIES ERROR:", str(e))
        return {"success": False, "message": str(e)}

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

        df_long = df.melt(
            id_vars=[date_col],
            value_vars=time_columns,
            var_name="time",
            value_name="consumption"
        )

        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.dropna(subset=["consumption"])

        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )

        df_long = df_long.dropna(subset=["timestamp"])
        df_agg = df_long.groupby("timestamp", as_index=False)["consumption"].sum()
        df_final = df_agg.copy()

        if df_final.empty:
            return {"success": False, "message": "No valid data after processing"}

        df_final["timestamp"] = df_final["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_final["consumption"] = df_final["consumption"].astype(float)

        records = df_final.to_dict(orient="records")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table("energy_data").insert(batch).execute()
            if hasattr(response, "error") and response.error:
                return {"success": False, "message": f"Supabase error: {response.error}"}

        return {"success": True, "rowsProcessed": len(records), "message": "Data stored in database"}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── ANALYTICS ────────────────────────────────────────────────

@app.get("/analytics")
def get_analytics():
    data = supabase.table("energy_data").select("*").range(0, 20000).execute().data

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
    df = parse_timestamps_naive(df)

    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["date"] = df["timestamp"].dt.date

    hourly_profile = build_hourly_profile(df)
    stats = build_stats(df)
    total_consumption = round(float(df["consumption"].sum()), 2)

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

        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"].astype(str),
            dayfirst=True,
            errors="coerce"
        )

        df_long = df_long.dropna(subset=["timestamp"])
        df_agg = df_long.groupby("timestamp", as_index=False)["consumption"].sum()
        df_final = df_agg.copy()

        if df_final.empty:
            return {"success": False, "message": "No valid data after processing"}

        df_final["timestamp"] = df_final["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_final["consumption"] = df_final["consumption"].astype(float)

        records = df_final.to_dict(orient="records")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table("gas_data").insert(batch).execute()
            if hasattr(response, "error") and response.error:
                return {"success": False, "message": f"Supabase error: {response.error}"}

        return {"success": True, "rowsProcessed": len(records), "message": "Gas data stored in database"}

    except Exception as e:
        print("GAS UPLOAD ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── GAS ANALYTICS ────────────────────────────────────────────

@app.get("/gas-analytics")
def get_gas_analytics():
    data = supabase.table("gas_data").select("*").range(0, 20000).execute().data

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

# ─── DEBUG: DATA SUMMARY ──────────────────────────────────────

@app.get("/debug/data-summary")
def debug_data_summary():
    try:
        data = supabase.table("energy_data").select("*").range(0, 20000).execute().data
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
        print("DEBUG SUMMARY ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── DEBUG: GAS DATA SUMMARY ──────────────────────────────────

@app.get("/debug/gas-summary")
def debug_gas_summary():
    try:
        data = supabase.table("gas_data").select("*").range(0, 20000).execute().data
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
