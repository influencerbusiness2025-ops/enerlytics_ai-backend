from fastapi import FastAPI, UploadFile, File, Query, BackgroundTasks, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from io import StringIO
from supabase import create_client
import httpx
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, List
import json
import os

# ─── SUPABASE CONFIG ──────────────────────────────────────────

SUPABASE_URL = "https://fopzbnloivgxzupxvhcr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvcHpibmxvaXZneHp1cHh2aGNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5Nzk5ODcsImV4cCI6MjA5MDU1NTk4N30.GC0Rs6N79vcXuyVBCqpyS5xH76sJ-Ea2CrY22gPyDMs"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── ANTHROPIC CONFIG ─────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-opus-4-5"

# ─── RATES ────────────────────────────────────────────────────

ELECTRICITY_RATE_GBP = 0.28
GAS_RATE_GBP         = 0.07

# ─── APP INIT ─────────────────────────────────────────────────

app = FastAPI()

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

class ReportRequest(BaseModel):
    report_type: str       # basic | ai_insights | full | premium_full
    date_from: str         # YYYY-MM-DD
    date_to: str           # YYYY-MM-DD
    period_type: str = "custom"  # monthly | annual | custom
    org_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str              # user | assistant
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    org_id: Optional[str] = None

# ─── TIER ENFORCEMENT ─────────────────────────────────────────

TIER_FEATURES = {
    "basic": [
        "dashboard", "analytics", "anomalies", "upload_data",
        "report_basic", "settings_sites",
    ],
    "standard": [
        "dashboard", "analytics", "anomalies", "upload_data",
        "ai_insights", "ai_recommendations", "weather_normalisation",
        "report_basic", "report_ai_insights", "report_full",
        "settings_sites", "multi_site",
    ],
    "premium": [
        "dashboard", "analytics", "anomalies", "upload_data",
        "ai_insights", "ai_recommendations", "ai_energy_analyst",
        "weather_normalisation",
        "report_basic", "report_ai_insights", "report_full", "report_premium_full",
        "settings_sites", "multi_site", "api_access",
    ],
}

REPORT_TIER_MAP = {
    "basic":         "basic",
    "ai_insights":   "standard",
    "full":          "standard",
    "premium_full":  "premium",
}

def get_org_tier(org_id: Optional[str]) -> str:
    """Return the tier for an organisation. Defaults to premium for dev."""
    if not org_id:
        return "premium"
    try:
        result = supabase.table("organisations").select("tier").eq("id", org_id).single().execute()
        return result.data["tier"] if result.data else "basic"
    except Exception:
        return "basic"

def check_feature_access(org_id: Optional[str], feature: str) -> bool:
    tier = get_org_tier(org_id)
    return feature in TIER_FEATURES.get(tier, [])

def require_feature(org_id: Optional[str], feature: str):
    if not check_feature_access(org_id, feature):
        tier = get_org_tier(org_id)
        required = REPORT_TIER_MAP.get(feature, "standard")
        raise HTTPException(
            status_code=403,
            detail={
                "error": "upgrade_required",
                "message": f"This feature requires {required} plan or above.",
                "current_tier": tier,
                "required_tier": required,
                "upgrade_url": "/pricing",
            }
        )

# ─── HELPERS ──────────────────────────────────────────────────

def parse_timestamps_naive(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def build_hourly_profile(df: pd.DataFrame) -> list:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    hourly_profile = []
    for h in range(24):
        hour_df = df[df["hour"] == h]
        if hour_df.empty:
            hourly_profile.append({"hour": f"{h:02d}:00", "average": 0, "weekday": 0, "weekend": 0})
            continue
        avg = hour_df["consumption"].mean()
        wd  = hour_df[~hour_df["is_weekend"]]["consumption"].mean() if not hour_df[~hour_df["is_weekend"]].empty else avg
        we  = hour_df[hour_df["is_weekend"]]["consumption"].mean()  if not hour_df[hour_df["is_weekend"]].empty  else avg
        hourly_profile.append({"hour": f"{h:02d}:00", "average": round(float(avg),2),
                                "weekday": round(float(wd),2), "weekend": round(float(we),2)})
    return hourly_profile

def build_stats(df: pd.DataFrame) -> dict:
    baseload = df["consumption"].quantile(0.1)
    peak = df["consumption"].max()
    avg  = df["consumption"].mean()
    return {"baseload": round(float(baseload),2), "peakDemand": round(float(peak),2),
            "loadFactor": round(float(avg/peak),2) if peak else 0, "avgDaily": round(float(avg*24),2)}

def auto_base_temp(lat: float, current_base: float) -> float:
    if current_base == 15.5 and abs(lat) < 23.5:
        return 24.0 if abs(lat) < 15 else 18.0
    return current_base

def resolve_mode(mode: str, lat: float, base_temp: float) -> str:
    if mode != "auto": return mode
    if base_temp >= 22 or abs(lat) < 15: return "cdd_only"
    if base_temp >= 18 or abs(lat) < 23.5: return "cdd_only"
    return "both"

async def fetch_degree_days(lat, lng, base_temp, start_date, end_date):
    url = (f"https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lng}&start_date={start_date}&end_date={end_date}"
           f"&daily=temperature_2m_mean&timezone=UTC")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url); resp.raise_for_status()
        data = resp.json()
    return {d: {"mean_temp": round(t,1), "hdd": round(max(0.0, base_temp-t),2), "cdd": round(max(0.0, t-base_temp),2)}
            for d, t in zip(data["daily"]["time"], data["daily"]["temperature_2m_mean"]) if t is not None}

def estimate_sensitivity(consumption, hdd, cdd, mode):
    n = len(consumption)
    if n < 7: return 0.0, 0.0, float(np.mean(consumption)) if consumption else 0.0
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

async def call_claude(prompt: str, max_tokens: int = 4000) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_MODEL, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]},
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

async def call_claude_messages(messages: list, system: str = "", max_tokens: int = 2000) -> str:
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set")
    body = {"model": ANTHROPIC_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system:
        body["system"] = system
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json=body,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

def build_energy_summary_for_ai() -> dict:
    elec_data = (supabase.table("energy_data").select("timestamp, consumption").range(0, 20000).execute().data) or []
    gas_data  = (supabase.table("gas_data").select("timestamp, consumption").range(0, 20000).execute().data) or []
    summary = {}

    if elec_data:
        df = pd.DataFrame(elec_data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        df["hour"] = df["timestamp"].dt.hour
        df["dow"]  = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["date"] = df["timestamp"].dt.date.astype(str)
        df["is_weekend"] = df["dow"] >= 5
        daily   = df.groupby("date")["consumption"].sum()
        monthly = df.groupby("month")["consumption"].sum()
        off_hours = df[df["hour"].isin([22,23,0,1,2,3,4,5,6])]
        on_hours  = df[~df["hour"].isin([22,23,0,1,2,3,4,5,6])]
        hourly_avg = df.groupby("hour")["consumption"].mean()
        weekday_avg = df[~df["is_weekend"]].groupby("date")["consumption"].sum().mean()
        weekend_avg = df[df["is_weekend"]].groupby("date")["consumption"].sum().mean()
        mom_change = None
        ms = monthly.sort_index()
        if len(ms) >= 2:
            last, prev = float(ms.iloc[-1]), float(ms.iloc[-2])
            mom_change = round((last - prev) / prev * 100, 1) if prev else None
        total_kwh = round(float(df["consumption"].sum()), 1)
        summary["electricity"] = {
            "total_kwh": total_kwh, "total_cost_gbp": round(total_kwh * ELECTRICITY_RATE_GBP, 2),
            "avg_daily_kwh": round(float(daily.mean()), 1),
            "peak_daily_kwh": round(float(daily.max()), 1),
            "min_daily_kwh": round(float(daily.min()), 1),
            "baseload_kwh": round(float(df["consumption"].quantile(0.1)), 2),
            "peak_demand_kwh": round(float(df["consumption"].max()), 2),
            "peak_hour": int(hourly_avg.idxmax()), "quiet_hour": int(hourly_avg.idxmin()),
            "avg_weekday_daily": round(float(weekday_avg), 1) if not np.isnan(weekday_avg) else 0,
            "avg_weekend_daily": round(float(weekend_avg), 1) if not np.isnan(weekend_avg) else 0,
            "off_hours_avg_kwh": round(float(off_hours["consumption"].mean()), 2) if not off_hours.empty else 0,
            "on_hours_avg_kwh": round(float(on_hours["consumption"].mean()), 2) if not on_hours.empty else 0,
            "off_hours_pct": round(float(off_hours["consumption"].sum() / df["consumption"].sum() * 100), 1) if total_kwh else 0,
            "month_on_month_pct": mom_change,
            "monthly_breakdown": {str(k): round(float(v), 1) for k, v in monthly.items()},
            "data_from": str(df["date"].min()), "data_to": str(df["date"].max()),
            "days_of_data": int(df["date"].nunique()),
        }

    if gas_data:
        gdf = pd.DataFrame(gas_data)
        gdf["consumption"] = pd.to_numeric(gdf["consumption"], errors="coerce")
        gdf = gdf.dropna(subset=["consumption"])
        gdf = parse_timestamps_naive(gdf)
        gdf["date"] = gdf["timestamp"].dt.date.astype(str)
        gdf["month"] = gdf["timestamp"].dt.month
        gas_daily   = gdf.groupby("date")["consumption"].sum()
        gas_monthly = gdf.groupby("month")["consumption"].sum()
        gas_total   = round(float(gdf["consumption"].sum()), 1)
        summary["gas"] = {
            "total_kwh": gas_total, "total_cost_gbp": round(gas_total * GAS_RATE_GBP, 2),
            "avg_daily_kwh": round(float(gas_daily.mean()), 1),
            "peak_daily_kwh": round(float(gas_daily.max()), 1),
            "monthly_breakdown": {str(k): round(float(v), 1) for k, v in gas_monthly.items()},
            "data_from": str(gdf["date"].min()), "data_to": str(gdf["date"].max()),
        }

    elec_cost = summary.get("electricity", {}).get("total_cost_gbp", 0)
    gas_cost  = summary.get("gas", {}).get("total_cost_gbp", 0)
    summary["combined"] = {
        "total_energy_kwh": round(summary.get("electricity",{}).get("total_kwh",0) + summary.get("gas",{}).get("total_kwh",0), 1),
        "total_cost_gbp": round(elec_cost + gas_cost, 2),
        "electricity_rate": f"£{ELECTRICITY_RATE_GBP}/kWh",
        "gas_rate": f"£{GAS_RATE_GBP}/kWh",
    }
    return summary

# ─── ROOT ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Enerlytics API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ─── TIER / FEATURE FLAGS ─────────────────────────────────────

@app.get("/tier")
def get_tier(org_id: Optional[str] = Query(default=None)):
    tier = get_org_tier(org_id)
    return {
        "tier": tier,
        "features": TIER_FEATURES.get(tier, []),
        "upgrade_url": "/pricing",
    }

@app.get("/feature-flags")
def get_feature_flags(org_id: Optional[str] = Query(default=None)):
    tier = get_org_tier(org_id)
    allowed = TIER_FEATURES.get(tier, [])
    all_features = set(f for features in TIER_FEATURES.values() for f in features)
    return {
        "tier": tier,
        "flags": {f: f in allowed for f in all_features},
    }

# ─── SITES CRUD ───────────────────────────────────────────────

@app.get("/analytics/sites")
def get_sites():
    try:
        result = (supabase.table("sites").select("id, name, lat, lng, timezone, base_temp, mode, address, building_type")
                  .eq("is_active", True).order("name").execute())
        return {"sites": result.data or []}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/analytics/sites")
def create_site(site: SiteCreate):
    try:
        base_temp = auto_base_temp(site.lat, site.base_temp)
        result = supabase.table("sites").insert({
            "name": site.name, "lat": site.lat, "lng": site.lng,
            "timezone": site.timezone, "base_temp": base_temp,
            "mode": site.mode, "address": site.address,
            "building_type": site.building_type, "is_active": True,
        }).execute()
        return {"success": True, "site": result.data[0] if result.data else None}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.put("/analytics/sites/{site_id}")
def update_site(site_id: str, site: SiteCreate):
    try:
        result = (supabase.table("sites").update({
            "name": site.name, "lat": site.lat, "lng": site.lng,
            "timezone": site.timezone, "base_temp": site.base_temp,
            "mode": site.mode, "address": site.address, "building_type": site.building_type,
        }).eq("id", site_id).execute())
        return {"success": True, "site": result.data[0] if result.data else None}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.delete("/analytics/sites/{site_id}")
def delete_site(site_id: str):
    try:
        supabase.table("sites").update({"is_active": False}).eq("id", site_id).execute()
        return {"success": True, "deleted": site_id}
    except Exception as e:
        return {"success": False, "message": str(e)}

# ─── WEATHER NORMALISED ───────────────────────────────────────

@app.get("/analytics/weather-normalised")
async def get_weather_normalised(
    site_id: str = Query(...), org_id: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None), end_date: Optional[str] = Query(default=None),
):
    require_feature(org_id, "weather_normalisation")
    try:
        site_result = (supabase.table("sites").select("*").eq("id", site_id).eq("is_active", True).single().execute())
        if not site_result.data:
            return {"success": False, "message": f"Site '{site_id}' not found"}
        site = site_result.data
        base_temp = site["base_temp"]
        mode = resolve_mode(site["mode"], site["lat"], base_temp)
        today = datetime.utcnow().date()
        if not end_date: end_date = str(today - timedelta(days=1))
        if not start_date: start_date = str(today - timedelta(days=31))
        data = (supabase.table("energy_data").select("timestamp, consumption")
                .gte("timestamp", start_date).lte("timestamp", end_date + "T23:59:59")
                .range(0, 20000).execute().data)
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
        actual_start, actual_end = daily_df["date"].min(), daily_df["date"].max()
        degree_days = await fetch_degree_days(site["lat"], site["lng"], base_temp, actual_start, actual_end)
        rows = []
        for _, row in daily_df.iterrows():
            dd = degree_days.get(row["date"], {"hdd": 0.0, "cdd": 0.0, "mean_temp": None})
            rows.append({"date": row["date"], "actual": float(row["actual"]),
                         "hdd": dd["hdd"], "cdd": dd["cdd"], "mean_temp": dd["mean_temp"]})
        beta_h, beta_c, baseload = estimate_sensitivity(
            [r["actual"] for r in rows], [r["hdd"] for r in rows], [r["cdd"] for r in rows], mode)
        results = []
        for r in rows:
            wi = round((r["hdd"]*beta_h) + (r["cdd"]*beta_c), 1)
            results.append({"date": r["date"], "actual": round(r["actual"],1),
                            "normalised": round(r["actual"]-wi,1), "weatherImpact": wi,
                            "hdd": r["hdd"], "cdd": r["cdd"], "meanTemp": r["mean_temp"]})
        ta = round(sum(r["actual"] for r in results), 1)
        tn = round(sum(r["normalised"] for r in results), 1)
        tw = round(sum(r["weatherImpact"] for r in results), 1)
        return {"site": {"id": site["id"], "name": site["name"], "baseTemp": base_temp, "mode": mode},
                "dateRange": {"start": actual_start, "end": actual_end},
                "coefficients": {"beta_h": round(beta_h,3), "beta_c": round(beta_c,3), "baseload": round(baseload,1)},
                "summary": {"totalActual": ta, "totalNormalised": tn, "totalWeatherImpact": tw,
                            "weatherImpactPct": round(tw/ta*100,1) if ta else 0},
                "daily": results}
    except httpx.RequestError as e:
        return {"success": False, "message": f"Weather API error: {str(e)}"}
    except Exception as e:
        return {"success": False, "message": str(e)}

# ─── ANOMALIES ────────────────────────────────────────────────

@app.get("/anomalies")
def get_anomalies(
    days: int = Query(default=90, ge=7, le=365),
    severity: Optional[str] = Query(default=None),
    anomaly_type: Optional[str] = Query(default=None),
):
    try:
        end_dt   = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)
        data = (supabase.table("energy_data").select("timestamp, consumption")
                .gte("timestamp", start_dt.strftime("%Y-%m-%dT%H:%M:%S"))
                .lte("timestamp", end_dt.strftime("%Y-%m-%dT%H:%M:%S"))
                .range(0, 20000).execute().data)
        if not data:
            return {"anomalies": [], "summary": {"total":0,"high":0,"medium":0,"low":0,"spikes":0,"drops":0},
                    "chartData": [], "avgDaily": 0, "heatmap": [[0]*24 for _ in range(7)], "totalScanned": 0}
        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["date"]         = df["timestamp"].dt.date.astype(str)
        df["hour"]         = df["timestamp"].dt.hour
        df["dow"]          = df["timestamp"].dt.dayofweek
        df["hour_of_week"] = df["dow"] * 24 + df["hour"]
        baseline = df.groupby("hour_of_week")["consumption"].median().to_dict()
        df["expected"] = df["hour_of_week"].map(baseline)
        std_map = df.groupby("hour_of_week")["consumption"].std().fillna(0).to_dict()
        df["std"] = df["hour_of_week"].map(std_map)
        anomalies = []
        for _, row in df.iterrows():
            expected, actual, std = row["expected"], row["consumption"], row["std"]
            if expected == 0: continue
            deviation_pct = ((actual - expected) / expected) * 100
            std_deviation = (actual - expected) / std if std > 0 else 0
            abs_std, abs_pct = abs(std_deviation), abs(deviation_pct)
            if abs_std >= 3.0 and abs_pct >= 20: pass
            elif abs_std >= 2.0 and abs_pct >= 30: pass
            elif abs_std >= 1.5 and abs_pct >= 50: pass
            else: continue
            a_type = "spike" if actual > expected else "drop"
            sev = "high" if (abs_std >= 3.0 or abs_pct >= 100) else "medium" if (abs_std >= 2.0 or abs_pct >= 50) else "low"
            anomalies.append({
                "timestamp": row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "date": row["date"], "hour": int(row["hour"]), "hourLabel": f"{int(row['hour']):02d}:00",
                "actual": round(float(actual),2), "expected": round(float(expected),2),
                "deviationPct": round(float(deviation_pct),1), "stdDeviations": round(float(std_deviation),2),
                "severity": sev, "type": a_type, "dow": int(row["dow"]),
                "dowLabel": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][int(row["dow"])],
            })
        filtered = list(anomalies)
        if severity: filtered = [a for a in filtered if a["severity"] == severity.lower()]
        if anomaly_type: filtered = [a for a in filtered if a["type"] == anomaly_type.lower()]
        sev_order = {"high":0,"medium":1,"low":2}
        filtered.sort(key=lambda x: (sev_order[x["severity"]], x["timestamp"]))
        summary = {"total": len(anomalies),
                   "high": sum(1 for a in anomalies if a["severity"]=="high"),
                   "medium": sum(1 for a in anomalies if a["severity"]=="medium"),
                   "low": sum(1 for a in anomalies if a["severity"]=="low"),
                   "spikes": sum(1 for a in anomalies if a["type"]=="spike"),
                   "drops": sum(1 for a in anomalies if a["type"]=="drop")}
        daily = df.groupby("date")["consumption"].sum().reset_index()
        anomaly_dates = {}
        for a in anomalies:
            d = a["date"]
            if d not in anomaly_dates or sev_order[a["severity"]] < sev_order[anomaly_dates[d]["sev"]]:
                anomaly_dates[d] = {"sev": a["severity"]}
        chart_data = [{"date": str(row["date"]), "consumption": round(float(row["consumption"]),2),
                       "hasAnomaly": str(row["date"]) in anomaly_dates,
                       "anomalySev": anomaly_dates[str(row["date"])]["sev"] if str(row["date"]) in anomaly_dates else None,
                       "anomalyCount": sum(1 for a in anomalies if a["date"] == str(row["date"]))}
                      for _, row in daily.iterrows()]
        heatmap = [[0]*24 for _ in range(7)]
        for a in anomalies: heatmap[a["dow"]][a["hour"]] += 1
        avg_daily = round(float(df.groupby("date")["consumption"].sum().mean()), 2)
        print(f"[anomalies] Scanned {len(df)} rows → {len(anomalies)} anomalies "
              f"(high={summary['high']}, medium={summary['medium']}, low={summary['low']})")
        return {"anomalies": filtered, "summary": summary, "chartData": chart_data,
                "avgDaily": avg_daily, "heatmap": heatmap, "totalScanned": len(df)}
    except Exception as e:
        print("ANOMALIES ERROR:", str(e))
        return {"success": False, "message": str(e)}

# ─── AI INSIGHTS ──────────────────────────────────────────────

async def generate_ai_insights(stats: dict) -> dict:
    elec, gas, comb = stats.get("electricity",{}), stats.get("gas",{}), stats.get("combined",{})
    prompt = f"""You are an expert energy analyst. Analyse this building energy data and return ONLY valid JSON.

ELECTRICITY: total={elec.get('total_kwh')}kWh, cost=£{elec.get('total_cost_gbp')}, avg_daily={elec.get('avg_daily_kwh')}kWh,
baseload={elec.get('baseload_kwh')}kWh/h, peak_hour={elec.get('peak_hour')}:00, off_hours_pct={elec.get('off_hours_pct')}%,
weekday_avg={elec.get('avg_weekday_daily')}kWh, weekend_avg={elec.get('avg_weekend_daily')}kWh,
month_on_month={elec.get('month_on_month_pct')}%, monthly={json.dumps(elec.get('monthly_breakdown',{}))},
period={elec.get('data_from')} to {elec.get('data_to')} ({elec.get('days_of_data')} days)

GAS: total={gas.get('total_kwh')}kWh, cost=£{gas.get('total_cost_gbp')}, avg_daily={gas.get('avg_daily_kwh')}kWh,
monthly={json.dumps(gas.get('monthly_breakdown',{}))}

COMBINED: total={comb.get('total_energy_kwh')}kWh, cost=£{comb.get('total_cost_gbp')}

Return this exact JSON structure:
{{
  "executive_summary": "2-3 sentences for business owner focusing on costs and opportunities",
  "insights": [
    {{"id":"slug","category":"baseload|peak_demand|off_hours|weekday_weekend|seasonal|gas|cost|trend",
      "title":"Short title","finding":"2-3 sentences with specific numbers",
      "implication":"Why this matters","severity":"high|medium|low|positive",
      "audience":["facilities","consultant","executive"],"metric":"key number","metric_label":"label"}}
  ],
  "recommendations": [
    {{"id":"slug","title":"Action title","action":"Specific step",
      "rationale":"Why based on data","saving_kwh_monthly":0,"saving_gbp_monthly":0,
      "effort":"low|medium|high","timeframe":"immediate|1_month|3_months|6_months",
      "payback_months":0,"category":"behavioural|controls|equipment|monitoring|procurement",
      "audience":["facilities","consultant","executive"],"priority":"quick_win|medium_term|long_term"}}
  ]
}}

Generate 5-8 insights and 5-7 recommendations. Hotel/hospitality context. Return ONLY JSON."""
    raw = await call_claude(prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())

async def run_ai_generation():
    print("[ai] Starting insight generation...")
    placeholder = supabase.table("ai_insights").insert({"status":"generating","generated_at":datetime.utcnow().isoformat()}).execute()
    row_id = placeholder.data[0]["id"] if placeholder.data else None
    try:
        stats = build_energy_summary_for_ai()
        if not stats.get("electricity") and not stats.get("gas"):
            raise ValueError("No energy data available")
        result = await generate_ai_insights(stats)
        data_from = stats.get("electricity",{}).get("data_from") or stats.get("gas",{}).get("data_from")
        data_to   = stats.get("electricity",{}).get("data_to")   or stats.get("gas",{}).get("data_to")
        payload = {"status":"complete","generated_at":datetime.utcnow().isoformat(),
                   "data_from":data_from,"data_to":data_to,
                   "executive_summary":result.get("executive_summary",""),
                   "insights":result.get("insights",[]),"recommendations":result.get("recommendations",[]),
                   "raw_stats":stats}
        if row_id: supabase.table("ai_insights").update(payload).eq("id", row_id).execute()
        else: supabase.table("ai_insights").insert(payload).execute()
        print(f"[ai] Done — {len(result.get('insights',[]))} insights, {len(result.get('recommendations',[]))} recommendations")
    except Exception as e:
        print(f"[ai] Failed: {e}")
        if row_id:
            supabase.table("ai_insights").update({"status":"error","error_message":str(e)}).eq("id",row_id).execute()

@app.get("/ai/insights")
def get_ai_insights(org_id: Optional[str] = Query(default=None)):
    require_feature(org_id, "ai_insights")
    try:
        result = (supabase.table("ai_insights").select("*").eq("status","complete")
                  .order("generated_at", desc=True).limit(1).execute())
        if not result.data:
            pending = (supabase.table("ai_insights").select("id,status,generated_at")
                       .eq("status","generating").order("generated_at",desc=True).limit(1).execute())
            if pending.data: return {"status":"generating","insights":None}
            return {"status":"empty","insights":None}
        row = result.data[0]
        return {"status":"complete","generatedAt":row["generated_at"],"dataFrom":row["data_from"],
                "dataTo":row["data_to"],"executiveSummary":row["executive_summary"],
                "insights":row["insights"] or [],"recommendations":row["recommendations"] or []}
    except HTTPException: raise
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.post("/ai/insights/generate")
async def trigger_ai_generation(background_tasks: BackgroundTasks, org_id: Optional[str] = Query(default=None)):
    require_feature(org_id, "ai_insights")
    if not ANTHROPIC_API_KEY:
        return {"success":False,"message":"ANTHROPIC_API_KEY not configured"}
    background_tasks.add_task(run_ai_generation)
    return {"success":True,"message":"Generation started. Poll GET /ai/insights for results."}

@app.get("/ai/insights/history")
def get_ai_insights_history():
    try:
        result = (supabase.table("ai_insights")
                  .select("id,generated_at,data_from,data_to,status,error_message")
                  .order("generated_at",desc=True).limit(10).execute())
        return {"history": result.data or []}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── AI ENERGY ANALYST (PREMIUM) ─────────────────────────────

@app.post("/ai/analyst/chat")
async def ai_analyst_chat(req: ChatRequest):
    require_feature(req.org_id, "ai_energy_analyst")
    try:
        # Build site context from energy data
        stats = build_energy_summary_for_ai()
        elec = stats.get("electricity", {})
        gas  = stats.get("gas", {})

        system_prompt = f"""You are Enerlytics AI — an expert energy analyst assistant with deep knowledge of this building's energy data.

BUILDING DATA CONTEXT:
Electricity: {elec.get('total_kwh','N/A')} kWh total, £{elec.get('total_cost_gbp','N/A')} cost
Period: {elec.get('data_from','N/A')} to {elec.get('data_to','N/A')} ({elec.get('days_of_data','N/A')} days)
Avg daily: {elec.get('avg_daily_kwh','N/A')} kWh | Peak hour: {elec.get('peak_hour','N/A')}:00
Baseload: {elec.get('baseload_kwh','N/A')} kWh/h | Off-hours share: {elec.get('off_hours_pct','N/A')}%
Weekday avg: {elec.get('avg_weekday_daily','N/A')} kWh | Weekend avg: {elec.get('avg_weekend_daily','N/A')} kWh
Month-on-month change: {elec.get('month_on_month_pct','N/A')}%
Monthly breakdown: {json.dumps(elec.get('monthly_breakdown',{}))}

Gas: {gas.get('total_kwh','N/A')} kWh total, £{gas.get('total_cost_gbp','N/A')} cost
Gas avg daily: {gas.get('avg_daily_kwh','N/A')} kWh

You can:
1. Answer questions specifically about THIS building's energy data using the context above
2. Answer general energy management, efficiency, and sustainability questions
3. Explain anomalies, patterns, and benchmarks
4. Provide calculations and cost estimates
5. Suggest specific actions based on the data

Be concise, specific, and use numbers from the data when relevant.
Format responses clearly — use bullet points for lists, bold for key numbers.
Always be helpful to facilities managers, energy consultants, and executives."""

        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        response = await call_claude_messages(messages, system=system_prompt, max_tokens=2000)

        # Save conversation to DB if org_id provided
        if req.org_id:
            all_messages = messages + [{"role":"assistant","content":response}]
            if req.conversation_id:
                supabase.table("ai_conversations").update({
                    "messages": all_messages,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", req.conversation_id).execute()
            else:
                # Auto-title from first user message
                title = req.messages[0].content[:60] + "..." if len(req.messages[0].content) > 60 else req.messages[0].content
                result = supabase.table("ai_conversations").insert({
                    "org_id": req.org_id, "title": title,
                    "messages": all_messages,
                }).execute()
                if result.data:
                    return {"response": response, "conversation_id": result.data[0]["id"]}

        return {"response": response, "conversation_id": req.conversation_id}

    except HTTPException: raise
    except Exception as e:
        print(f"[ai-analyst] ERROR: {e}")
        return {"success": False, "message": str(e)}

@app.get("/ai/analyst/conversations")
def get_conversations(org_id: str = Query(...)):
    require_feature(org_id, "ai_energy_analyst")
    try:
        result = (supabase.table("ai_conversations").select("id, title, created_at, updated_at")
                  .eq("org_id", org_id).order("updated_at", desc=True).limit(20).execute())
        return {"conversations": result.data or []}
    except HTTPException: raise
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.get("/ai/analyst/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    try:
        result = supabase.table("ai_conversations").select("*").eq("id", conversation_id).single().execute()
        return {"conversation": result.data}
    except Exception as e:
        return {"success": False, "message": str(e)}

# ─── REPORTS ──────────────────────────────────────────────────

def build_report_data(date_from: str, date_to: str, report_type: str) -> dict:
    """Build the data payload for any report type."""

    # ── Electricity data ──────────────────────────────────────
    elec_data = (supabase.table("energy_data").select("timestamp, consumption")
                 .gte("timestamp", date_from).lte("timestamp", date_to + "T23:59:59")
                 .range(0, 20000).execute().data) or []

    # ── Gas data ──────────────────────────────────────────────
    gas_data = (supabase.table("gas_data").select("timestamp, consumption")
                .gte("timestamp", date_from).lte("timestamp", date_to + "T23:59:59")
                .range(0, 20000).execute().data) or []

    report = {"report_type": report_type, "date_from": date_from, "date_to": date_to,
              "generated_at": datetime.utcnow().isoformat()}

    # ── Electricity stats ─────────────────────────────────────
    if elec_data:
        df = pd.DataFrame(elec_data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        df["date"]  = df["timestamp"].dt.date.astype(str)
        df["hour"]  = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        df["dow"]   = df["timestamp"].dt.dayofweek

        daily   = df.groupby("date")["consumption"].sum()
        monthly = df.groupby("month")["consumption"].sum()
        total_kwh = round(float(df["consumption"].sum()), 2)

        report["electricity"] = {
            "total_kwh": total_kwh,
            "total_cost_gbp": round(total_kwh * ELECTRICITY_RATE_GBP, 2),
            "avg_daily_kwh": round(float(daily.mean()), 2),
            "peak_daily_kwh": round(float(daily.max()), 2),
            "baseload_kwh": round(float(df["consumption"].quantile(0.1)), 2),
            "peak_demand_kwh": round(float(df["consumption"].max()), 2),
            "hourly_profile": build_hourly_profile(df),
            "daily_breakdown": [{"date": d, "kwh": round(float(v),2), "cost": round(float(v)*ELECTRICITY_RATE_GBP,2)}
                                 for d, v in daily.items()],
            "monthly_breakdown": [{"month": int(m), "kwh": round(float(v),2), "cost": round(float(v)*ELECTRICITY_RATE_GBP,2)}
                                   for m, v in monthly.items()],
        }

    # ── Gas stats ─────────────────────────────────────────────
    if gas_data:
        gdf = pd.DataFrame(gas_data)
        gdf["consumption"] = pd.to_numeric(gdf["consumption"], errors="coerce")
        gdf = gdf.dropna(subset=["consumption"])
        gdf = parse_timestamps_naive(gdf)
        gdf["date"]  = gdf["timestamp"].dt.date.astype(str)
        gdf["month"] = gdf["timestamp"].dt.month
        gas_daily   = gdf.groupby("date")["consumption"].sum()
        gas_monthly = gdf.groupby("month")["consumption"].sum()
        gas_total   = round(float(gdf["consumption"].sum()), 2)
        report["gas"] = {
            "total_kwh": gas_total,
            "total_cost_gbp": round(gas_total * GAS_RATE_GBP, 2),
            "avg_daily_kwh": round(float(gas_daily.mean()), 2),
            "daily_breakdown": [{"date": d, "kwh": round(float(v),2), "cost": round(float(v)*GAS_RATE_GBP,2)}
                                 for d, v in gas_daily.items()],
            "monthly_breakdown": [{"month": int(m), "kwh": round(float(v),2), "cost": round(float(v)*GAS_RATE_GBP,2)}
                                   for m, v in gas_monthly.items()],
        }

    # ── Anomaly summary ───────────────────────────────────────
    if report_type in ("basic", "full", "premium_full") and elec_data:
        report["anomaly_summary"] = {
            "note": "See /anomalies endpoint for full anomaly detection details",
            "period": f"{date_from} to {date_to}"
        }

    # ── Combined totals ───────────────────────────────────────
    elec_cost = report.get("electricity", {}).get("total_cost_gbp", 0)
    gas_cost  = report.get("gas", {}).get("total_cost_gbp", 0)
    elec_kwh  = report.get("electricity", {}).get("total_kwh", 0)
    gas_kwh   = report.get("gas", {}).get("total_kwh", 0)
    report["combined"] = {
        "total_kwh":  round(elec_kwh + gas_kwh, 2),
        "total_cost_gbp": round(elec_cost + gas_cost, 2),
        "electricity_share_pct": round(elec_kwh / (elec_kwh + gas_kwh) * 100, 1) if (elec_kwh + gas_kwh) else 0,
        "gas_share_pct": round(gas_kwh / (elec_kwh + gas_kwh) * 100, 1) if (elec_kwh + gas_kwh) else 0,
    }

    # ── AI insights (standard+ reports) ──────────────────────
    if report_type in ("ai_insights", "full", "premium_full"):
        ai_result = (supabase.table("ai_insights").select("executive_summary,insights,recommendations")
                     .eq("status","complete").order("generated_at",desc=True).limit(1).execute())
        if ai_result.data:
            row = ai_result.data[0]
            report["ai_insights"] = {
                "executive_summary": row["executive_summary"],
                "insights": row["insights"] or [],
                "recommendations": row["recommendations"] or [],
            }

    return report


@app.post("/reports/generate")
def generate_report(req: ReportRequest):
    """Generate a report and save to Supabase."""
    # Check tier access
    feature_map = {
        "basic":         "report_basic",
        "ai_insights":   "report_ai_insights",
        "full":          "report_full",
        "premium_full":  "report_premium_full",
    }
    require_feature(req.org_id, feature_map.get(req.report_type, "report_basic"))

    try:
        report_data = build_report_data(req.date_from, req.date_to, req.report_type)

        # Title generation
        titles = {
            "basic":        "Basic Energy Report",
            "ai_insights":  "AI Insights Report",
            "full":         "Full Energy Report",
            "premium_full": "Premium Full Report",
        }
        title = f"{titles.get(req.report_type, 'Report')} — {req.date_from} to {req.date_to}"

        result = supabase.table("reports").insert({
            "org_id":      req.org_id,
            "report_type": req.report_type,
            "title":       title,
            "date_from":   req.date_from,
            "date_to":     req.date_to,
            "period_type": req.period_type,
            "status":      "complete",
            "data":        report_data,
        }).execute()

        return {
            "success": True,
            "report_id": result.data[0]["id"] if result.data else None,
            "title": title,
            "data": report_data,
        }
    except HTTPException: raise
    except Exception as e:
        print("REPORT GENERATE ERROR:", str(e))
        return {"success": False, "message": str(e)}


@app.get("/reports")
def list_reports(org_id: Optional[str] = Query(default=None)):
    """List all reports for an organisation."""
    try:
        query = supabase.table("reports").select("id,report_type,title,date_from,date_to,period_type,status,created_at")
        if org_id: query = query.eq("org_id", org_id)
        result = query.order("created_at", desc=True).limit(50).execute()
        return {"reports": result.data or []}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/reports/{report_id}")
def get_report(report_id: str):
    """Get a specific report by ID."""
    try:
        result = supabase.table("reports").select("*").eq("id", report_id).single().execute()
        if not result.data:
            return {"success": False, "message": "Report not found"}
        return {"report": result.data}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.delete("/reports/{report_id}")
def delete_report(report_id: str):
    try:
        supabase.table("reports").delete().eq("id", report_id).execute()
        return {"success": True, "deleted": report_id}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/reports/preview/{report_type}")
def preview_report(
    report_type: str,
    date_from: str = Query(...),
    date_to: str = Query(...),
    org_id: Optional[str] = Query(default=None),
):
    """Preview report data without saving."""
    feature_map = {"basic":"report_basic","ai_insights":"report_ai_insights",
                   "full":"report_full","premium_full":"report_premium_full"}
    require_feature(org_id, feature_map.get(report_type, "report_basic"))
    try:
        data = build_report_data(date_from, date_to, report_type)
        return {"success": True, "data": data}
    except HTTPException: raise
    except Exception as e:
        return {"success": False, "message": str(e)}

# ─── UPLOAD CSV ───────────────────────────────────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), index_col=None)
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if not date_col: return {"success":False,"message":"No date column found"}
        time_columns = [col for col in df.columns if ":" in col]
        if not time_columns: return {"success":False,"message":"No time columns found"}
        df_long = df.melt(id_vars=[date_col], value_vars=time_columns, var_name="time", value_name="consumption")
        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.dropna(subset=["consumption"])
        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"].astype(str), dayfirst=True, errors="coerce")
        df_long = df_long.dropna(subset=["timestamp"])
        df_agg = df_long.groupby("timestamp", as_index=False)["consumption"].sum()
        if df_agg.empty: return {"success":False,"message":"No valid data after processing"}
        df_agg["timestamp"] = df_agg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_agg["consumption"] = df_agg["consumption"].astype(float)
        records = df_agg.to_dict(orient="records")
        for i in range(0, len(records), 500):
            supabase.table("energy_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Data stored in database"}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── ANALYTICS ────────────────────────────────────────────────

@app.get("/analytics")
def get_analytics():
    data = supabase.table("energy_data").select("*").range(0, 20000).execute().data
    if not data:
        return {"stats":{"baseload":0,"peakDemand":0,"loadFactor":0,"avgDaily":0},
                "hourlyProfile":[],"daily":[],"totalConsumption":0,"heatmap":[[0.0]*24 for _ in range(7)]}
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
        hourly_values = [round(float(day_df[day_df["hour"]==h]["consumption"].sum()),2) for h in range(24)]
        daily_breakdown.append({"date":str(date_val),"consumption":round(float(row["consumption"]),2),"hourly":hourly_values})
    heatmap = [[0.0]*24 for _ in range(7)]
    counts  = [[0]*24   for _ in range(7)]
    for _, row in df.iterrows():
        d, h = row["timestamp"].dayofweek, row["timestamp"].hour
        heatmap[d][h] += row["consumption"]; counts[d][h] += 1
    for d in range(7):
        for h in range(24):
            if counts[d][h] > 0: heatmap[d][h] = round(heatmap[d][h]/counts[d][h], 2)
    return {"stats":stats,"hourlyProfile":hourly_profile,"daily":daily_breakdown,
            "totalConsumption":total_consumption,"heatmap":heatmap}

@app.get("/analytics/hourly-profile/{year}")
def get_hourly_profile_by_year(year: int):
    try:
        data = (supabase.table("energy_data").select("timestamp, consumption")
                .gte("timestamp",f"{year}-01-01").lte("timestamp",f"{year}-12-31T23:59:59").execute().data)
        if not data:
            return {"hourlyProfile":[{"hour":f"{h:02d}:00","average":0,"weekday":0,"weekend":0} for h in range(24)]}
        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        return {"hourlyProfile": build_hourly_profile(df)}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── UPLOAD GAS / GAS ANALYTICS ───────────────────────────────

@app.post("/upload-gas-data")
async def upload_gas_data(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), index_col=None)
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if not date_col: return {"success":False,"message":"No date column found"}
        time_columns = [col for col in df.columns if ":" in col]
        if not time_columns: return {"success":False,"message":"No time columns found"}
        df_long = df.melt(id_vars=[date_col], value_vars=time_columns, var_name="time", value_name="consumption")
        df_long["consumption"] = pd.to_numeric(df_long["consumption"], errors="coerce")
        df_long = df_long.dropna(subset=["consumption"])
        df_long["timestamp"] = pd.to_datetime(
            df_long[date_col].astype(str) + " " + df_long["time"].astype(str), dayfirst=True, errors="coerce")
        df_long = df_long.dropna(subset=["timestamp"])
        df_agg = df_long.groupby("timestamp", as_index=False)["consumption"].sum()
        if df_agg.empty: return {"success":False,"message":"No valid data"}
        df_agg["timestamp"] = df_agg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_agg["consumption"] = df_agg["consumption"].astype(float)
        records = df_agg.to_dict(orient="records")
        for i in range(0, len(records), 500):
            supabase.table("gas_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Gas data stored"}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/gas-analytics")
def get_gas_analytics():
    data = supabase.table("gas_data").select("*").range(0, 20000).execute().data
    if not data:
        return {"stats":{"baseload":0,"peakDemand":0,"loadFactor":0,"avgDaily":0},"hourlyProfile":[],"daily":[],"totalConsumption":0}
    df = pd.DataFrame(data)
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df = df.dropna(subset=["consumption"])
    df = parse_timestamps_naive(df)
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date")["consumption"].sum().reset_index()
    return {"stats":build_stats(df),"hourlyProfile":build_hourly_profile(df),
            "daily":[{"date":str(r["date"]),"consumption":round(float(r["consumption"]),2)} for _,r in daily.iterrows()],
            "totalConsumption":round(float(df["consumption"].sum()),2)}

# ─── DEBUG ────────────────────────────────────────────────────

@app.get("/debug/data-summary")
def debug_data_summary():
    try:
        data = supabase.table("energy_data").select("*").range(0, 20000).execute().data
        if not data: return {"rowCount":0}
        df = pd.DataFrame(data)
        df = parse_timestamps_naive(df)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df["date"] = df["timestamp"].dt.date
        hour_dist = df.groupby(df["timestamp"].dt.hour)["consumption"].count().to_dict()
        unique_dates = df["date"].nunique()
        total = float(df["consumption"].sum())
        return {"rowCount":len(df),"dateRange":{"earliest":str(df["timestamp"].min()),"latest":str(df["timestamp"].max())},
                "totalConsumption":round(total,2),"avgConsumption":round(float(df["consumption"].mean()),2),
                "minConsumption":round(float(df["consumption"].min()),2),"maxConsumption":round(float(df["consumption"].max()),2),
                "peakDemand":round(float(df["consumption"].max()),2),
                "avgDaily":round(total/unique_dates,2) if unique_dates else 0,"hourDistribution":hour_dist}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/debug/gas-summary")
def debug_gas_summary():
    try:
        data = supabase.table("gas_data").select("*").range(0, 20000).execute().data
        if not data: return {"rowCount":0}
        df = pd.DataFrame(data)
        df = parse_timestamps_naive(df)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df["date"] = df["timestamp"].dt.date
        hour_dist = df.groupby(df["timestamp"].dt.hour)["consumption"].count().to_dict()
        unique_dates = df["date"].nunique()
        total = float(df["consumption"].sum())
        return {"rowCount":len(df),"dateRange":{"earliest":str(df["timestamp"].min()),"latest":str(df["timestamp"].max())},
                "totalConsumption":round(total,2),"avgConsumption":round(float(df["consumption"].mean()),2),
                "minConsumption":round(float(df["consumption"].min()),2),"maxConsumption":round(float(df["consumption"].max()),2),
                "peakDemand":round(float(df["consumption"].max()),2),
                "avgDaily":round(total/unique_dates,2) if unique_dates else 0,"hourDistribution":hour_dist}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── DELETE DATA ──────────────────────────────────────────────

@app.delete("/delete-data")
def delete_data():
    try:
        supabase.table("energy_data").delete().gt("id","00000000-0000-0000-0000-000000000000").execute()
        return {"success":True,"message":"All energy data deleted"}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.delete("/delete-gas-data")
def delete_gas_data():
    try:
        supabase.table("gas_data").delete().gt("id","00000000-0000-0000-0000-000000000000").execute()
        return {"success":True,"message":"All gas data deleted"}
    except Exception as e:
        return {"success":False,"message":str(e)}
