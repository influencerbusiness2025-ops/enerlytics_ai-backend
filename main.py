from fastapi import FastAPI, UploadFile, File, Query, BackgroundTasks, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from io import StringIO
from supabase import create_client
import httpx
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import json
import os
import hmac
import hashlib

# ─── CONFIG ───────────────────────────────────────────────────

SUPABASE_URL         = "https://fopzbnloivgxzupxvhcr.supabase.co"
SUPABASE_KEY         = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvcHpibmxvaXZneHp1cHh2aGNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5Nzk5ODcsImV4cCI6MjA5MDU1NTk4N30.GC0Rs6N79vcXuyVBCqpyS5xH76sJ-Ea2CrY22gPyDMs"
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", SUPABASE_KEY)
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL      = "claude-opus-4-5"
STRIPE_SECRET_KEY       = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET   = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_BASIC_MONTHLY    = os.environ.get("STRIPE_PRICE_BASIC_MONTHLY", "")
STRIPE_PRICE_BASIC_ANNUAL     = os.environ.get("STRIPE_PRICE_BASIC_ANNUAL", "")
STRIPE_PRICE_STANDARD_MONTHLY = os.environ.get("STRIPE_PRICE_STANDARD_MONTHLY", "")
STRIPE_PRICE_STANDARD_ANNUAL  = os.environ.get("STRIPE_PRICE_STANDARD_ANNUAL", "")
STRIPE_PRICE_PREMIUM_MONTHLY  = os.environ.get("STRIPE_PRICE_PREMIUM_MONTHLY", "")
STRIPE_PRICE_PREMIUM_ANNUAL   = os.environ.get("STRIPE_PRICE_PREMIUM_ANNUAL", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://ai.effictraenergy.co.uk")
ELECTRICITY_RATE_GBP = 0.28
GAS_RATE_GBP         = 0.07

supabase         = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─── APP ──────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ─── MODELS ───────────────────────────────────────────────────

class SiteCreate(BaseModel):
    name: str; lat: float; lng: float
    timezone: str = "UTC"; base_temp: float = 15.5
    mode: str = "auto"; address: str = ""; building_type: str = "commercial"

class ReportRequest(BaseModel):
    report_type: str; date_from: str; date_to: str
    period_type: str = "custom"; org_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str; content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    org_id: Optional[str] = None

class CheckoutRequest(BaseModel):
    plan: str; billing_period: str; org_id: str
    success_url: Optional[str] = None; cancel_url: Optional[str] = None

# ─── BMS MODELS ───────────────────────────────────────────────

class EquipmentCreate(BaseModel):
    site_id: str
    name: str
    category: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    bms_ref: Optional[str] = None
    is_active: bool = True

class EquipmentUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    bms_ref: Optional[str] = None
    is_active: Optional[bool] = None

class ParameterCreate(BaseModel):
    equipment_id: str
    parameter_name: str
    parameter_type: str
    unit: Optional[str] = None
    bms_tag: Optional[str] = None

# ─── TIER CONFIG ──────────────────────────────────────────────

TIER_FEATURES = {
    "trial": {
        "dashboard": True, "analytics": True, "anomalies": True, "upload_data": True,
        "ai_insights": True, "ai_recommendations": True, "ai_energy_analyst": True,
        "ai_senior_consultant": False, "weather_normalisation": True,
        "report_basic": True, "report_ai_insights": True, "report_full": True,
        "report_premium_full": True, "settings_sites": True, "multi_site": False,
        "api_access": False, "bms_parameters": True,
    },
    "basic": {
        "dashboard": True, "analytics": True, "anomalies": True, "upload_data": True,
        "ai_insights": False, "ai_recommendations": False, "ai_energy_analyst": False,
        "ai_senior_consultant": False, "weather_normalisation": False,
        "report_basic": True, "report_ai_insights": False, "report_full": False,
        "report_premium_full": False, "settings_sites": True, "multi_site": False,
        "api_access": False, "bms_parameters": False,
    },
    "standard": {
        "dashboard": True, "analytics": True, "anomalies": True, "upload_data": True,
        "ai_insights": True, "ai_recommendations": True, "ai_energy_analyst": False,
        "ai_senior_consultant": False, "weather_normalisation": True,
        "report_basic": True, "report_ai_insights": True, "report_full": True,
        "report_premium_full": False, "settings_sites": True, "multi_site": True,
        "api_access": False, "bms_parameters": False,
    },
    "premium": {
        "dashboard": True, "analytics": True, "anomalies": True, "upload_data": True,
        "ai_insights": True, "ai_recommendations": True, "ai_energy_analyst": True,
        "ai_senior_consultant": False, "weather_normalisation": True,
        "report_basic": True, "report_ai_insights": True, "report_full": True,
        "report_premium_full": True, "settings_sites": True, "multi_site": True,
        "api_access": True, "bms_parameters": True,
    },
    "enterprise": {
        "dashboard": True, "analytics": True, "anomalies": True, "upload_data": True,
        "ai_insights": True, "ai_recommendations": True, "ai_energy_analyst": True,
        "ai_senior_consultant": True, "weather_normalisation": True,
        "report_basic": True, "report_ai_insights": True, "report_full": True,
        "report_premium_full": True, "settings_sites": True, "multi_site": True,
        "api_access": True, "bms_parameters": True,
    },
}

FEATURE_REQUIRED_TIER = {
    "ai_insights":           "standard",
    "ai_recommendations":    "standard",
    "weather_normalisation": "standard",
    "report_ai_insights":    "standard",
    "report_full":           "standard",
    "multi_site":            "standard",
    "ai_energy_analyst":     "premium",
    "report_premium_full":   "premium",
    "api_access":            "premium",
    "bms_parameters":        "premium",
    "ai_senior_consultant":  "enterprise",
}

STRIPE_PRICE_MAP = {
    ("basic","monthly"):    STRIPE_PRICE_BASIC_MONTHLY,
    ("basic","annual"):     STRIPE_PRICE_BASIC_ANNUAL,
    ("standard","monthly"): STRIPE_PRICE_STANDARD_MONTHLY,
    ("standard","annual"):  STRIPE_PRICE_STANDARD_ANNUAL,
    ("premium","monthly"):  STRIPE_PRICE_PREMIUM_MONTHLY,
    ("premium","annual"):   STRIPE_PRICE_PREMIUM_ANNUAL,
}

# ─── AUTH HELPERS ─────────────────────────────────────────────

def get_user_from_token(authorization):
    if not authorization or not authorization.startswith("Bearer "): return None
    token = authorization.replace("Bearer ", "")
    try:
        result = supabase_service.auth.get_user(token)
        return result.user if result else None
    except Exception: return None

def get_org_for_user(auth_id):
    try:
        result = (supabase.table("users").select("org_id, role, organisations(*)")
                  .eq("auth_id", auth_id).single().execute())
        if not result.data: return None
        org = result.data.get("organisations", {})
        if org.get("tier") == "trial":
            from datetime import timezone
            trial_expires = org.get("trial_ends_at")
            if trial_expires:
                expires_dt = datetime.fromisoformat(trial_expires.replace("Z", "+00:00"))
                if expires_dt < datetime.now(timezone.utc):
                    supabase.table("organisations").update({"tier":"basic"}).eq("id",org["id"]).execute()
                    org["tier"] = "basic"
        return org
    except Exception as e:
        print(f"[auth] error: {e}"); return None

def get_effective_tier(org): return org.get("tier","basic") if org else "basic"

def require_auth(authorization):
    auth_user = get_user_from_token(authorization)
    if not auth_user:
        raise HTTPException(status_code=401, detail={"error":"unauthorized","message":"Please log in."})
    org = get_org_for_user(auth_user.id)
    return auth_user, org

def get_org_tier_by_id(org_id):
    if not org_id: return "basic"
    try:
        result = supabase.table("organisations").select("tier,trial_ends_at").eq("id",org_id).single().execute()
        if not result.data: return "basic"
        org = result.data
        if org.get("tier") == "trial":
            from datetime import timezone
            trial_expires = org.get("trial_ends_at")
            if trial_expires:
                expires_dt = datetime.fromisoformat(trial_expires.replace("Z","+00:00"))
                if expires_dt < datetime.now(timezone.utc):
                    supabase.table("organisations").update({"tier":"basic"}).eq("id",org_id).execute()
                    return "basic"
        tier = org.get("tier", "basic")
        if tier not in TIER_FEATURES: tier = "basic"
        return tier
    except Exception: return "basic"

def require_feature(org_id, feature):
    tier = get_org_tier_by_id(org_id)
    features = TIER_FEATURES.get(tier, {})
    if not features.get(feature, False):
        required = FEATURE_REQUIRED_TIER.get(feature, "standard")
        raise HTTPException(status_code=403, detail={
            "error":"upgrade_required", "message":f"This feature requires the {required} plan.",
            "current_tier":tier, "required_tier":required, "upgrade_url":f"{FRONTEND_URL}/pricing"
        })

def resolve_tier(authorization, org_id):
    if authorization and authorization.startswith("Bearer "):
        try:
            _, org = require_auth(authorization)
            return get_effective_tier(org), org
        except HTTPException: pass
    return get_org_tier_by_id(org_id), None

def require_feature_jwt(authorization, org_id, feature):
    tier, _ = resolve_tier(authorization, org_id)
    features = TIER_FEATURES.get(tier, {})
    if not features.get(feature, False):
        required = FEATURE_REQUIRED_TIER.get(feature, "standard")
        raise HTTPException(status_code=403, detail={
            "error":"upgrade_required", "message":f"This feature requires the {required} plan.",
            "current_tier":tier, "required_tier":required, "upgrade_url":f"{FRONTEND_URL}/pricing"
        })

# ─── ENERGY HELPERS ───────────────────────────────────────────

def parse_timestamps_naive(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"])

def build_hourly_profile(df):
    df = df.copy(); df["hour"]=df["timestamp"].dt.hour
    df["is_weekend"]=df["timestamp"].dt.dayofweek>=5
    profile=[]
    for h in range(24):
        hdf=df[df["hour"]==h]
        if hdf.empty: profile.append({"hour":f"{h:02d}:00","average":0,"weekday":0,"weekend":0}); continue
        avg=hdf["consumption"].mean()
        wd=hdf[~hdf["is_weekend"]]["consumption"].mean() if not hdf[~hdf["is_weekend"]].empty else avg
        we=hdf[hdf["is_weekend"]]["consumption"].mean() if not hdf[hdf["is_weekend"]].empty else avg
        profile.append({"hour":f"{h:02d}:00","average":round(float(avg),2),
                        "weekday":round(float(wd),2),"weekend":round(float(we),2)})
    return profile

def build_stats(df):
    peak=df["consumption"].max(); avg=df["consumption"].mean()
    return {"baseload":round(float(df["consumption"].quantile(0.1)),2),
            "peakDemand":round(float(peak),2),
            "loadFactor":round(float(avg/peak),2) if peak else 0,
            "avgDaily":round(float(avg*24),2)}

def auto_base_temp(lat,base):
    if base==15.5 and abs(lat)<23.5: return 24.0 if abs(lat)<15 else 18.0
    return base

def resolve_mode(mode,lat,base_temp):
    if mode!="auto": return mode
    if base_temp>=22 or abs(lat)<15: return "cdd_only"
    if base_temp>=18 or abs(lat)<23.5: return "cdd_only"
    return "both"

async def fetch_degree_days(lat,lng,base_temp,start_date,end_date):
    url=(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}"
         f"&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean&timezone=UTC")
    async with httpx.AsyncClient(timeout=15.0) as client:
        r=await client.get(url); r.raise_for_status(); data=r.json()
    return {d:{"mean_temp":round(t,1),"hdd":round(max(0.0,base_temp-t),2),"cdd":round(max(0.0,t-base_temp),2)}
            for d,t in zip(data["daily"]["time"],data["daily"]["temperature_2m_mean"]) if t is not None}

def estimate_sensitivity(consumption,hdd,cdd,mode):
    n=len(consumption)
    if n<7: return 0.0,0.0,float(np.mean(consumption)) if consumption else 0.0
    y=np.array(consumption,dtype=float)
    if mode=="cdd_only":
        X=np.column_stack([np.ones(n),np.array(cdd)]); c=np.linalg.lstsq(X,y,rcond=None)[0]
        return 0.0,max(0.0,float(c[1])),float(c[0])
    elif mode=="hdd_only":
        X=np.column_stack([np.ones(n),np.array(hdd)]); c=np.linalg.lstsq(X,y,rcond=None)[0]
        return max(0.0,float(c[1])),0.0,float(c[0])
    else:
        X=np.column_stack([np.ones(n),np.array(hdd),np.array(cdd)]); c=np.linalg.lstsq(X,y,rcond=None)[0]
        return max(0.0,float(c[1])),max(0.0,float(c[2])),float(c[0])

async def call_claude(prompt,max_tokens=4000):
    if not ANTHROPIC_API_KEY: raise ValueError("ANTHROPIC_API_KEY not set")
    async with httpx.AsyncClient(timeout=120.0) as client:
        r=await client.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":ANTHROPIC_MODEL,"max_tokens":max_tokens,"messages":[{"role":"user","content":prompt}]})
        r.raise_for_status(); return r.json()["content"][0]["text"]

def build_energy_summary_for_ai(org_id=None):
    elec_q=supabase.table("energy_data").select("timestamp,consumption").range(0,20000)
    gas_q=supabase.table("gas_data").select("timestamp,consumption").range(0,20000)
    if org_id: elec_q=elec_q.eq("org_id",org_id); gas_q=gas_q.eq("org_id",org_id)
    elec_data=elec_q.execute().data or []
    gas_data=gas_q.execute().data or []
    summary={}
    if elec_data:
        df=pd.DataFrame(elec_data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        df["hour"]=df["timestamp"].dt.hour; df["dow"]=df["timestamp"].dt.dayofweek
        df["month"]=df["timestamp"].dt.month; df["date"]=df["timestamp"].dt.date.astype(str)
        df["is_weekend"]=df["dow"]>=5
        daily=df.groupby("date")["consumption"].sum(); monthly=df.groupby("month")["consumption"].sum()
        off_hours=df[df["hour"].isin([22,23,0,1,2,3,4,5,6])]
        hourly_avg=df.groupby("hour")["consumption"].mean()
        wd_avg=df[~df["is_weekend"]].groupby("date")["consumption"].sum().mean()
        we_avg=df[df["is_weekend"]].groupby("date")["consumption"].sum().mean()
        ms=monthly.sort_index(); mom=None
        if len(ms)>=2:
            last,prev=float(ms.iloc[-1]),float(ms.iloc[-2])
            mom=round((last-prev)/prev*100,1) if prev else None
        total_kwh=round(float(df["consumption"].sum()),1)
        summary["electricity"]={
            "total_kwh":total_kwh,"total_cost_gbp":round(total_kwh*ELECTRICITY_RATE_GBP,2),
            "avg_daily_kwh":round(float(daily.mean()),1),"peak_daily_kwh":round(float(daily.max()),1),
            "min_daily_kwh":round(float(daily.min()),1),
            "baseload_kwh":round(float(df["consumption"].quantile(0.1)),2),
            "peak_demand_kwh":round(float(df["consumption"].max()),2),
            "peak_hour":int(hourly_avg.idxmax()),"quiet_hour":int(hourly_avg.idxmin()),
            "avg_weekday_daily":round(float(wd_avg),1) if not np.isnan(wd_avg) else 0,
            "avg_weekend_daily":round(float(we_avg),1) if not np.isnan(we_avg) else 0,
            "off_hours_avg_kwh":round(float(off_hours["consumption"].mean()),2) if not off_hours.empty else 0,
            "off_hours_pct":round(float(off_hours["consumption"].sum()/df["consumption"].sum()*100),1) if total_kwh else 0,
            "month_on_month_pct":mom,
            "monthly_breakdown":{str(k):round(float(v),1) for k,v in monthly.items()},
            "data_from":str(df["date"].min()),"data_to":str(df["date"].max()),
            "days_of_data":int(df["date"].nunique())
        }
    if gas_data:
        gdf=pd.DataFrame(gas_data); gdf["consumption"]=pd.to_numeric(gdf["consumption"],errors="coerce")
        gdf=gdf.dropna(subset=["consumption"]); gdf=parse_timestamps_naive(gdf)
        gdf["date"]=gdf["timestamp"].dt.date.astype(str); gdf["month"]=gdf["timestamp"].dt.month
        gas_daily=gdf.groupby("date")["consumption"].sum()
        gas_monthly=gdf.groupby("month")["consumption"].sum()
        gas_total=round(float(gdf["consumption"].sum()),1)
        summary["gas"]={
            "total_kwh":gas_total,"total_cost_gbp":round(gas_total*GAS_RATE_GBP,2),
            "avg_daily_kwh":round(float(gas_daily.mean()),1),
            "peak_daily_kwh":round(float(gas_daily.max()),1),
            "monthly_breakdown":{str(k):round(float(v),1) for k,v in gas_monthly.items()},
            "data_from":str(gdf["date"].min()),"data_to":str(gdf["date"].max())
        }
    ec=summary.get("electricity",{}).get("total_cost_gbp",0)
    gc=summary.get("gas",{}).get("total_cost_gbp",0)
    summary["combined"]={
        "total_energy_kwh":round(summary.get("electricity",{}).get("total_kwh",0)+summary.get("gas",{}).get("total_kwh",0),1),
        "total_cost_gbp":round(ec+gc,2)
    }
    return summary

# ─── BMS HELPERS ──────────────────────────────────────────────

def build_bms_context_for_ai(site_id: Optional[str] = None, days: int = 7) -> str:
    """
    Fetch recent BMS readings and format as context string for AI prompts.
    Returns empty string if no BMS data exists.
    """
    try:
        eq_query = supabase.table("equipment").select("id,name,category,manufacturer,model").eq("is_active", True)
        if site_id:
            eq_query = eq_query.eq("site_id", site_id)
        equipment_rows = eq_query.execute().data or []
        if not equipment_rows:
            return ""

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        context_lines = [f"BMS EQUIPMENT PARAMETERS (last {days} days):"]

        for eq in equipment_rows:
            eq_id = eq["id"]
            params = (supabase.table("equipment_parameters")
                      .select("id,parameter_name,parameter_type,unit")
                      .eq("equipment_id", eq_id).execute().data) or []
            if not params:
                continue

            eq_label = f"{eq['category'].upper()} — {eq['name']}"
            if eq.get("manufacturer"):
                eq_label += f" ({eq['manufacturer']} {eq.get('model','')})"
            eq_lines = [f"\n{eq_label}"]

            for param in params:
                readings = (supabase.table("equipment_readings")
                            .select("recorded_at,value,value_text")
                            .eq("parameter_id", param["id"])
                            .gte("recorded_at", cutoff)
                            .order("recorded_at", desc=True)
                            .limit(100).execute().data) or []

                if not readings:
                    eq_lines.append(f"  {param['parameter_name']} ({param['parameter_type']}): no recent data")
                    continue

                ptype = param["parameter_type"]
                unit = param.get("unit") or ""

                if ptype == "on_off":
                    on_count = sum(1 for r in readings if str(r.get("value","")).strip() in ("1","1.0") or str(r.get("value_text","")).lower() in ("on","true","yes"))
                    total = len(readings)
                    on_pct = round(on_count / total * 100) if total else 0
                    latest_val = readings[0].get("value_text") or readings[0].get("value")
                    eq_lines.append(f"  {param['parameter_name']}: currently {latest_val} | ON {on_pct}% of readings ({on_count}/{total})")

                elif ptype in ("flow_temp", "return_temp", "setpoint_temp"):
                    vals = [float(r["value"]) for r in readings if r.get("value") is not None]
                    if vals:
                        eq_lines.append(f"  {param['parameter_name']}: latest {vals[0]}{unit} | avg {round(sum(vals)/len(vals),1)}{unit} | range {round(min(vals),1)}–{round(max(vals),1)}{unit}")

                elif ptype == "run_hours":
                    vals = [float(r["value"]) for r in readings if r.get("value") is not None]
                    if vals:
                        eq_lines.append(f"  {param['parameter_name']}: {vals[0]}{unit} (latest reading)")

                elif ptype == "mode":
                    from collections import Counter
                    modes = [str(r.get("value_text") or r.get("value","")).strip() for r in readings if r.get("value_text") or r.get("value")]
                    if modes:
                        breakdown = ", ".join(f"{m}: {c}" for m,c in Counter(modes).most_common())
                        eq_lines.append(f"  {param['parameter_name']}: currently {modes[0]} | distribution: {breakdown}")

                elif ptype == "fault_alarm":
                    active = [r for r in readings if str(r.get("value","")).strip() in ("1","1.0") or str(r.get("value_text","")).lower() in ("fault","alarm","true","active")]
                    if active:
                        eq_lines.append(f"  *** FAULT/ALARM ACTIVE — {param['parameter_name']}: last triggered {active[0]['recorded_at']} ({len(active)} occurrences) ***")
                    else:
                        eq_lines.append(f"  {param['parameter_name']}: no active faults")

                else:
                    vals = [float(r["value"]) for r in readings if r.get("value") is not None]
                    if vals:
                        eq_lines.append(f"  {param['parameter_name']}: {vals[0]}{unit} (latest)")

            context_lines.extend(eq_lines)

        return "\n".join(context_lines) if len(context_lines) > 1 else ""

    except Exception as e:
        print(f"[bms_context] Error: {e}")
        return ""


def parse_bms_csv(contents: bytes, parameter_id: str) -> List[dict]:
    """
    Parse a BMS CSV upload. Flexible column detection and robust date parsing.
    Expects: timestamp column + value column (+ optional status/text column).
    Handles all common date formats globally.
    """
    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    df.columns = [c.strip().lower() for c in df.columns]

    ts_col = next((c for c in df.columns if any(k in c for k in ("time","date","timestamp"))), None)
    if not ts_col:
        raise ValueError("CSV must have a timestamp/datetime column")

    val_col = next((c for c in df.columns if c in ("value","val","reading","data")), None)
    if not val_col:
        val_col = next((c for c in df.columns if c != ts_col), None)
    if not val_col:
        raise ValueError("CSV must have a value column")

    val_text_col = next((c for c in df.columns if any(k in c for k in ("text","status","state")) and c != val_col), None)

    def parse_timestamp_flexible(ts_str):
        """
        Try multiple date formats in order of specificity.
        Handles ISO 8601, UK/EU (dd/mm/yyyy), US (mm/dd/yyyy), epoch, and more.
        """
        if pd.isna(ts_str) or str(ts_str).strip() == "":
            return pd.NaT

        ts_str = str(ts_str).strip()

        # Try pandas auto-detect first (handles ISO 8601, most standard formats)
        try:
            result = pd.to_datetime(ts_str, infer_datetime_format=True)
            if not pd.isna(result):
                return result
        except Exception:
            pass

        # Explicit format attempts — ordered by global prevalence
        formats = [
            # ISO variants
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            # UK/EU: day first
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%d.%m.%Y",
            # US: month first
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%m-%d-%Y %H:%M:%S",
            "%m-%d-%Y %H:%M",
            # BMS-specific common exports
            "%d %b %Y %H:%M:%S",
            "%d %b %Y %H:%M",
            "%b %d %Y %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
        ]

        for fmt in formats:
            try:
                result = datetime.strptime(ts_str, fmt)
                return result
            except ValueError:
                continue

        # Last resort: unix epoch (integer or float seconds)
        try:
            epoch = float(ts_str)
            return datetime.utcfromtimestamp(epoch)
        except (ValueError, OSError):
            pass

        return pd.NaT

    df["_ts"] = df[ts_col].apply(parse_timestamp_flexible)
    df = df.dropna(subset=["_ts"])

    if df.empty:
        raise ValueError(
            "Could not parse any timestamps. Please ensure first row is headers and "
            "timestamps are in a standard format (e.g. 2024-01-15 08:00, 15/01/2024 08:00, or 01/15/2024 08:00)."
        )

    df["_val"] = pd.to_numeric(df[val_col], errors="coerce")

    records = []
    for _, row in df.iterrows():
        ts = row["_ts"]
        # Normalise to naive UTC isoformat string
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        rec = {
            "parameter_id": parameter_id,
            "recorded_at": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "value": float(row["_val"]) if not pd.isna(row["_val"]) else None,
            "value_text": str(row[val_text_col]).strip() if val_text_col and not pd.isna(row.get(val_text_col)) else None,
        }
        records.append(rec)
    return records


# ─── RAG TOOLS ────────────────────────────────────────────────

ANALYST_TOOLS = [
    {"name":"get_hourly_data",
     "description":"Get actual hourly electricity consumption for a specific date or range.",
     "input_schema":{"type":"object","properties":{
         "start_date":{"type":"string","description":"Start date YYYY-MM-DD"},
         "end_date":{"type":"string","description":"End date YYYY-MM-DD"},
         "hour":{"type":"integer","description":"Specific hour 0-23","minimum":0,"maximum":23}},
         "required":["start_date","end_date"]}},
    {"name":"get_daily_summary",
     "description":"Get daily total electricity consumption and cost for a date range.",
     "input_schema":{"type":"object","properties":{
         "start_date":{"type":"string"},"end_date":{"type":"string"}},
         "required":["start_date","end_date"]}},
    {"name":"get_anomalies",
     "description":"Get detected energy anomalies (spikes/drops) for a period.",
     "input_schema":{"type":"object","properties":{
         "start_date":{"type":"string"},"end_date":{"type":"string"},
         "severity":{"type":"string","enum":["high","medium","low"]}},
         "required":["start_date","end_date"]}},
    {"name":"get_peak_hours",
     "description":"Find highest/lowest electricity consumption hours.",
     "input_schema":{"type":"object","properties":{
         "start_date":{"type":"string"},"end_date":{"type":"string"},
         "top_n":{"type":"integer","default":5}},
         "required":["start_date","end_date"]}},
    {"name":"compare_periods",
     "description":"Compare electricity consumption between two time periods.",
     "input_schema":{"type":"object","properties":{
         "period1_start":{"type":"string"},"period1_end":{"type":"string"},
         "period2_start":{"type":"string"},"period2_end":{"type":"string"}},
         "required":["period1_start","period1_end","period2_start","period2_end"]}},
    {"name":"get_monthly_stats",
     "description":"Get monthly electricity consumption breakdown for a year.",
     "input_schema":{"type":"object","properties":{
         "year":{"type":"integer","description":"Year e.g. 2024"}},
         "required":["year"]}},
    {"name":"get_gas_data",
     "description":"Get gas consumption data for a date range. Returns daily totals and summary stats.",
     "input_schema":{"type":"object","properties":{
         "start_date":{"type":"string","description":"Start date YYYY-MM-DD"},
         "end_date":{"type":"string","description":"End date YYYY-MM-DD"}},
         "required":["start_date","end_date"]}},
    {"name":"get_site_equipment",
     "description":"Get all BMS equipment for the site, grouped by category (heating/cooling/ventilation/pumps). Shows equipment names, categories, and whether any faults are active.",
     "input_schema":{"type":"object","properties":{
         "category":{"type":"string","description":"Filter by category: heating, cooling, ventilation, pumps, lighting, other. Leave empty for all."}},
         "required":[]}},
    {"name":"get_equipment_readings",
     "description":"Get BMS parameter readings for a specific piece of equipment. Provide equipment_name (e.g. AHU-01) or equipment_id. Returns all parameters with recent values: temperatures, on/off status, run hours, valve positions, fault alarms.",
     "input_schema":{"type":"object","properties":{
         "equipment_name":{"type":"string","description":"Equipment name e.g. AHU-01, Chiller-1, Boiler"},
         "start_date":{"type":"string","description":"Start date YYYY-MM-DD"},
         "end_date":{"type":"string","description":"End date YYYY-MM-DD"}},
         "required":["start_date","end_date"]}},
    {"name":"get_active_faults",
     "description":"Get all currently active BMS fault alarms across all equipment on the site. Always call this first when asked about equipment problems or faults.",
     "input_schema":{"type":"object","properties":{},"required":[]}},
]

def execute_tool(tool_name, tool_input, org_id=None):
    try:
        if tool_name=="get_hourly_data":
            return _tool_get_hourly_data(tool_input["start_date"],tool_input["end_date"],tool_input.get("hour"),org_id=org_id)
        elif tool_name=="get_daily_summary":
            return _tool_get_daily_summary(tool_input["start_date"],tool_input["end_date"],org_id=org_id)
        elif tool_name=="get_anomalies":
            return _tool_get_anomalies(tool_input["start_date"],tool_input["end_date"],tool_input.get("severity"),org_id=org_id)
        elif tool_name=="get_peak_hours":
            return _tool_get_peak_hours(tool_input["start_date"],tool_input["end_date"],tool_input.get("top_n",5),org_id=org_id)
        elif tool_name=="compare_periods":
            return _tool_compare_periods(tool_input["period1_start"],tool_input["period1_end"],
                                         tool_input["period2_start"],tool_input["period2_end"],org_id=org_id)
        elif tool_name=="get_monthly_stats":
            return _tool_get_monthly_stats(tool_input["year"],org_id=org_id)
        elif tool_name=="get_gas_data":
            return _tool_get_gas_data(tool_input["start_date"],tool_input["end_date"],org_id=org_id)
        elif tool_name=="get_site_equipment":
            return _tool_get_site_equipment(tool_input.get("category"))
        elif tool_name=="get_equipment_readings":
            return _tool_get_equipment_readings(
                tool_input.get("equipment_name",""),
                tool_input["start_date"],tool_input["end_date"])
        elif tool_name=="get_active_faults":
            return _tool_get_active_faults()
        return f"Unknown tool: {tool_name}"
    except Exception as e: return f"Tool error: {str(e)}"


def _tool_get_gas_data(start_date, end_date, org_id=None):
    q=(supabase.table("gas_data").select("timestamp,consumption")
          .gte("timestamp",start_date).lte("timestamp",end_date+"T23:59:59")
          .range(0,20000))
    if org_id: q=q.eq("org_id",org_id)
    data=q.execute().data or []
    if not data: return f"No gas data for {start_date} to {end_date}"
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df["date"]=df["timestamp"].dt.date.astype(str)
    daily=df.groupby("date")["consumption"].sum()
    total=round(float(df["consumption"].sum()),2)
    cost=round(total*GAS_RATE_GBP,2)
    result=[f"Gas data {start_date} to {end_date}:",
            f"Total: {total} kWh (£{cost})",
            f"Daily avg: {round(float(daily.mean()),2)} kWh",
            f"Peak: {daily.idxmax()} — {round(float(daily.max()),2)} kWh",
            "Daily breakdown:"]
    for date,kwh in daily.items():
        result.append(f"  {date}: {round(float(kwh),2)} kWh (£{round(float(kwh)*GAS_RATE_GBP,2)})")
    return "\n".join(result)


def _tool_get_site_equipment(category=None):
    try:
        query = supabase.table("equipment").select("id,name,category,manufacturer,model,bms_ref").eq("is_active",True)
        if category:
            query = query.eq("category", category)
        equipment_rows = query.order("category").order("name").execute().data or []
        if not equipment_rows:
            return "No equipment configured on this site yet."
        cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
        result = [f"Site equipment ({len(equipment_rows)} items):"]
        by_cat = {}
        for eq in equipment_rows:
            cat = eq["category"]
            if cat not in by_cat: by_cat[cat] = []
            # Check for active faults
            params = (supabase.table("equipment_parameters")
                      .select("id,parameter_type")
                      .eq("equipment_id", eq["id"])
                      .eq("parameter_type","fault_alarm").execute().data) or []
            has_fault = False
            for p in params:
                latest = (supabase.table("equipment_readings")
                          .select("value,value_text")
                          .eq("parameter_id",p["id"])
                          .order("recorded_at",desc=True).limit(1).execute().data)
                if latest:
                    v = str(latest[0].get("value","")).strip()
                    vt = str(latest[0].get("value_text","")).lower()
                    if v in ("1","1.0") or vt in ("fault","alarm","true","active"):
                        has_fault = True
            fault_flag = " *** FAULT ACTIVE ***" if has_fault else ""
            by_cat[cat].append(f"  - {eq['name']}{' ('+eq['manufacturer']+')' if eq.get('manufacturer') else ''}{fault_flag} [id:{eq['id']}]")
        for cat, items in by_cat.items():
            result.append(f"\n{cat.upper()}:")
            result.extend(items)
        return "\n".join(result)
    except Exception as e:
        return f"Error fetching equipment: {e}"


def _tool_get_equipment_readings(equipment_name, start_date, end_date):
    try:
        # Find equipment by name
        query = supabase.table("equipment").select("id,name,category").eq("is_active",True)
        if equipment_name:
            # Try exact match first, then ilike
            exact = query.eq("name", equipment_name).execute().data
            if exact:
                equipment_rows = exact
            else:
                equipment_rows = (supabase.table("equipment").select("id,name,category")
                                  .eq("is_active",True)
                                  .ilike("name", f"%{equipment_name}%").execute().data) or []
        else:
            equipment_rows = query.execute().data or []

        if not equipment_rows:
            return f"No equipment found matching '{equipment_name}'. Use get_site_equipment to see available equipment."

        result = []
        for eq in equipment_rows[:3]:  # Limit to 3 equipment max
            result.append(f"\n{eq['category'].upper()} — {eq['name']}:")
            params = (supabase.table("equipment_parameters")
                      .select("id,parameter_name,parameter_type,unit")
                      .eq("equipment_id",eq["id"]).execute().data) or []
            if not params:
                result.append("  No parameters configured.")
                continue
            for param in params:
                readings = (supabase.table("equipment_readings")
                            .select("recorded_at,value,value_text")
                            .eq("parameter_id",param["id"])
                            .gte("recorded_at",start_date)
                            .lte("recorded_at",end_date+"T23:59:59")
                            .order("recorded_at",desc=True)
                            .limit(200).execute().data) or []
                if not readings:
                    result.append(f"  {param['parameter_name']} ({param['parameter_type']}): no data in range")
                    continue
                unit = param.get("unit") or ""
                ptype = param["parameter_type"]
                latest = readings[0]
                latest_val = latest.get("value_text") or latest.get("value")
                if ptype in ("flow_temp","return_temp","setpoint_temp","sensor","Htg_Vlv_pos","Clg_Vlv_Pos","Flow_Pressure_SP","Run_Speed"):
                    vals = [float(r["value"]) for r in readings if r.get("value") is not None]
                    if vals:
                        result.append(f"  {param['parameter_name']}: latest={vals[0]}{unit} avg={round(sum(vals)/len(vals),1)}{unit} min={round(min(vals),1)}{unit} max={round(max(vals),1)}{unit} ({len(vals)} readings)")
                elif ptype == "on_off":
                    on = sum(1 for r in readings if str(r.get("value","")).strip() in ("1","1.0") or str(r.get("value_text","")).lower() in ("on","true"))
                    result.append(f"  {param['parameter_name']}: currently {latest_val} | ON {round(on/len(readings)*100)}% of {len(readings)} readings")
                elif ptype == "fault_alarm":
                    faults = [r for r in readings if str(r.get("value","")).strip() in ("1","1.0") or str(r.get("value_text","")).lower() in ("fault","alarm","true","active")]
                    if faults:
                        result.append(f"  *** {param['parameter_name']}: FAULT ACTIVE — {len(faults)} fault events, last at {faults[0]['recorded_at']} ***")
                    else:
                        result.append(f"  {param['parameter_name']}: no faults in period")
                elif ptype == "run_hours":
                    vals = [float(r["value"]) for r in readings if r.get("value") is not None]
                    if vals:
                        result.append(f"  {param['parameter_name']}: {vals[0]}{unit} (latest)")
                elif ptype == "mode":
                    from collections import Counter
                    modes = [str(r.get("value_text") or r.get("value","")).strip() for r in readings if r.get("value_text") or r.get("value")]
                    if modes:
                        dist = ", ".join(f"{m}:{c}" for m,c in Counter(modes).most_common())
                        result.append(f"  {param['parameter_name']}: currently {modes[0]} | distribution: {dist}")
                else:
                    result.append(f"  {param['parameter_name']}: {latest_val}{unit} (latest)")
        return "\n".join(result)
    except Exception as e:
        return f"Error fetching equipment readings: {e}"


def _tool_get_active_faults():
    try:
        equipment_rows = (supabase.table("equipment").select("id,name,category")
                          .eq("is_active",True).execute().data) or []
        if not equipment_rows:
            return "No equipment configured on this site."
        cutoff = (datetime.utcnow() - timedelta(days=1)).isoformat()
        active_faults = []
        for eq in equipment_rows:
            fault_params = (supabase.table("equipment_parameters")
                            .select("id,parameter_name")
                            .eq("equipment_id",eq["id"])
                            .eq("parameter_type","fault_alarm").execute().data) or []
            for param in fault_params:
                latest = (supabase.table("equipment_readings")
                          .select("recorded_at,value,value_text")
                          .eq("parameter_id",param["id"])
                          .gte("recorded_at",cutoff)
                          .order("recorded_at",desc=True).limit(1).execute().data)
                if latest:
                    v = str(latest[0].get("value","")).strip()
                    vt = str(latest[0].get("value_text","")).lower()
                    if v in ("1","1.0") or vt in ("fault","alarm","true","active"):
                        active_faults.append(
                            f"  *** FAULT: {eq['category'].upper()} — {eq['name']} | {param['parameter_name']} | Last triggered: {latest[0]['recorded_at']} ***"
                        )
        if not active_faults:
            return "No active faults detected across all equipment in the last 24 hours."
        return "ACTIVE FAULTS:\n" + "\n".join(active_faults)
    except Exception as e:
        return f"Error checking faults: {e}"

def _tool_get_hourly_data(start_date,end_date,hour=None,org_id=None):
    q=(supabase.table("energy_data").select("timestamp,consumption")
          .gte("timestamp",start_date).lte("timestamp",end_date+"T23:59:59")
          .order("timestamp").range(0,500))
    if org_id: q=q.eq("org_id",org_id)
    data=q.execute().data or []
    if not data: return f"No data found for {start_date} to {end_date}"
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df["hour_of_day"]=df["timestamp"].dt.hour
    if hour is not None:
        df=df[df["hour_of_day"]==hour]
        if df.empty: return f"No data for {hour:02d}:00 between {start_date} and {end_date}"
    result=[f"Hourly data {start_date} to {end_date}"+(f" at {hour:02d}:00" if hour is not None else ""),
            f"Total: {round(df['consumption'].sum(),2)} kWh | Avg: {round(df['consumption'].mean(),2)} kWh",
            f"Peak: {round(df['consumption'].max(),2)} kWh at {df.loc[df['consumption'].idxmax(),'timestamp']}",
            f"Lowest: {round(df['consumption'].min(),2)} kWh at {df.loc[df['consumption'].idxmin(),'timestamp']}"]
    if len(df)<=48:
        result.append("Readings:")
        for _,row in df.iterrows():
            result.append(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: {round(row['consumption'],2)} kWh")
    return "\n".join(result)

def _tool_get_daily_summary(start_date,end_date,org_id=None):
    q=(supabase.table("energy_data").select("timestamp,consumption")
          .gte("timestamp",start_date).lte("timestamp",end_date+"T23:59:59")
          .range(0,20000))
    if org_id: q=q.eq("org_id",org_id)
    data=q.execute().data or []
    if not data: return f"No data for {start_date} to {end_date}"
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df["date"]=df["timestamp"].dt.date.astype(str)
    daily=df.groupby("date")["consumption"].sum().reset_index()
    daily["cost"]=(daily["consumption"]*ELECTRICITY_RATE_GBP).round(2)
    daily["consumption"]=daily["consumption"].round(2)
    result=[f"Daily summary {start_date} to {end_date}:",
            f"Total: {round(daily['consumption'].sum(),2)} kWh (£{round(daily['cost'].sum(),2)})",
            f"Daily avg: {round(daily['consumption'].mean(),2)} kWh",
            f"Peak: {daily.loc[daily['consumption'].idxmax(),'date']} — {daily['consumption'].max()} kWh",
            f"Lowest: {daily.loc[daily['consumption'].idxmin(),'date']} — {daily['consumption'].min()} kWh","Days:"]
    for _,row in daily.iterrows():
        result.append(f"  {row['date']}: {row['consumption']} kWh (£{row['cost']})")
    return "\n".join(result)

def _tool_get_anomalies(start_date,end_date,severity=None,org_id=None):
    q=(supabase.table("energy_data").select("timestamp,consumption")
          .gte("timestamp",start_date).lte("timestamp",end_date+"T23:59:59")
          .range(0,20000))
    if org_id: q=q.eq("org_id",org_id)
    data=q.execute().data or []
    if not data: return f"No data for {start_date} to {end_date}"
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df=df.sort_values("timestamp").reset_index(drop=True)
    df["hour"]=df["timestamp"].dt.hour; df["dow"]=df["timestamp"].dt.dayofweek
    df["hour_of_week"]=df["dow"]*24+df["hour"]
    baseline=df.groupby("hour_of_week")["consumption"].median().to_dict()
    df["expected"]=df["hour_of_week"].map(baseline)
    std_map=df.groupby("hour_of_week")["consumption"].std().fillna(0).to_dict()
    df["std"]=df["hour_of_week"].map(std_map)
    anomalies=[]
    for _,row in df.iterrows():
        expected,actual,std=row["expected"],row["consumption"],row["std"]
        if expected==0: continue
        dev_pct=((actual-expected)/expected)*100
        std_dev=(actual-expected)/std if std>0 else 0
        abs_std,abs_pct=abs(std_dev),abs(dev_pct)
        if abs_std>=3.0 and abs_pct>=20: pass
        elif abs_std>=2.0 and abs_pct>=30: pass
        elif abs_std>=1.5 and abs_pct>=50: pass
        else: continue
        a_type="spike" if actual>expected else "drop"
        sev="high" if (abs_std>=3.0 or abs_pct>=100) else "medium" if (abs_std>=2.0 or abs_pct>=50) else "low"
        if severity and sev!=severity: continue
        anomalies.append({"timestamp":row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "actual":round(float(actual),2),"expected":round(float(expected),2),
            "deviation_pct":round(float(dev_pct),1),"severity":sev,"type":a_type})
    if not anomalies: return f"No anomalies between {start_date} and {end_date}"
    result=[f"Found {len(anomalies)} anomalies:"]
    for a in anomalies[:20]:
        result.append(f"  [{a['severity'].upper()}] {a['timestamp']} — {a['type']}: {a['actual']} kWh (expected {a['expected']}, {a['deviation_pct']:+.1f}%)")
    if len(anomalies)>20: result.append(f"  ... and {len(anomalies)-20} more")
    return "\n".join(result)

def _tool_get_peak_hours(start_date,end_date,top_n=5,org_id=None):
    q=(supabase.table("energy_data").select("timestamp,consumption")
          .gte("timestamp",start_date).lte("timestamp",end_date+"T23:59:59")
          .range(0,20000))
    if org_id: q=q.eq("org_id",org_id)
    data=q.execute().data or []
    if not data: return f"No data for {start_date} to {end_date}"
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df=df.sort_values("consumption",ascending=False)
    result=[f"Top {top_n} peak readings:"]
    for _,row in df.head(top_n).iterrows():
        result.append(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: {round(row['consumption'],2)} kWh")
    result.append(f"Top {top_n} lowest readings:")
    for _,row in df.tail(top_n).iterrows():
        result.append(f"  {row['timestamp'].strftime('%Y-%m-%d %H:%M')}: {round(row['consumption'],2)} kWh")
    return "\n".join(result)

def _tool_compare_periods(p1_start,p1_end,p2_start,p2_end,org_id=None):
    def get_stats(start,end):
        q=(supabase.table("energy_data").select("timestamp,consumption")
              .gte("timestamp",start).lte("timestamp",end+"T23:59:59").range(0,20000))
        if org_id: q=q.eq("org_id",org_id)
        data=q.execute().data or []
        if not data: return None
        df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        df["date"]=df["timestamp"].dt.date.astype(str)
        daily=df.groupby("date")["consumption"].sum()
        return {"total":round(float(df["consumption"].sum()),2),"daily_avg":round(float(daily.mean()),2),
                "peak":round(float(df["consumption"].max()),2),"days":len(daily),
                "cost":round(float(df["consumption"].sum())*ELECTRICITY_RATE_GBP,2)}
    p1=get_stats(p1_start,p1_end); p2=get_stats(p2_start,p2_end)
    if not p1 or not p2: return "Insufficient data for one or both periods"
    change_pct=round((p2["total"]-p1["total"])/p1["total"]*100,1) if p1["total"] else 0
    return "\n".join([
        f"Period 1 ({p1_start} to {p1_end}, {p1['days']} days): {p1['total']} kWh (£{p1['cost']}) | avg {p1['daily_avg']} kWh/day",
        f"Period 2 ({p2_start} to {p2_end}, {p2['days']} days): {p2['total']} kWh (£{p2['cost']}) | avg {p2['daily_avg']} kWh/day",
        f"Change: {change_pct:+.1f}% | Cost diff: £{abs(round(p2['cost']-p1['cost'],2))} {'more' if p2['cost']>p1['cost'] else 'less'}"])

def _tool_get_monthly_stats(year,org_id=None):
    q=(supabase.table("energy_data").select("timestamp,consumption")
          .gte("timestamp",f"{year}-01-01").lte("timestamp",f"{year}-12-31T23:59:59")
          .range(0,20000))
    if org_id: q=q.eq("org_id",org_id)
    data=q.execute().data or []
    if not data: return f"No data for {year}"
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df["month"]=df["timestamp"].dt.month
    monthly=df.groupby("month")["consumption"].sum()
    months={1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    result=[f"Monthly stats for {year}:"]; total=0
    for m,kwh in monthly.items():
        cost=round(float(kwh)*ELECTRICITY_RATE_GBP,2); total+=float(kwh)
        result.append(f"  {months[m]}: {round(float(kwh),2)} kWh (£{cost})")
    result.append(f"Annual total: {round(total,2)} kWh (£{round(total*ELECTRICITY_RATE_GBP,2)})")
    if monthly.any():
        result.append(f"Peak: {months[monthly.idxmax()]} | Lowest: {months[monthly.idxmin()]}")
    return "\n".join(result)

# ─── AI ANALYST SYSTEM PROMPTS ────────────────────────────────

def build_data_analyst_prompt(elec, gas, today, bms_context=""):
    bms_section = f"\n{bms_context}\n" if bms_context else "\nBMS DATA: No equipment parameters uploaded yet.\n"
    return f"""You are Effictra AI Data Analyst — an intelligent energy data analyst embedded in this building's energy monitoring platform.

Today's date: {today}

YOUR ROLE:
You analyse this building's energy data, compare against industry benchmarks for similar buildings, and provide practical operational recommendations. You do NOT provide legal, compliance, or strategic consultancy advice — for that, users should upgrade to the Enterprise plan or contact Effictra Energy directly.

YOUR EXPERTISE:
- Analysing energy consumption patterns (hourly, daily, monthly, seasonal)
- Benchmarking against CIBSE TM46 and Carbon Trust standards for similar buildings
- Identifying inefficiencies: high baseload, off-hours waste, peak demand issues
- Operational best practices for HVAC, lighting, and building management
- Correlating BMS equipment parameters (temperatures, on/off status, run hours, faults) with energy patterns
- Understanding of building types: hotels, offices, retail, hospitals, industrial
- Basic carbon footprint calculations from energy data
- Simple ROI calculations for common energy efficiency measures

THIS BUILDING'S DATA:
Electricity: {elec.get('total_kwh','N/A')} kWh total | £{elec.get('total_cost_gbp','N/A')} cost
Period: {elec.get('data_from','N/A')} to {elec.get('data_to','N/A')} ({elec.get('days_of_data','N/A')} days)
Avg daily: {elec.get('avg_daily_kwh','N/A')} kWh | Peak hour: {elec.get('peak_hour','N/A')}:00
Baseload: {elec.get('baseload_kwh','N/A')} kWh/h | Peak demand: {elec.get('peak_demand_kwh','N/A')} kWh/h
Off-hours (22:00-06:00): {elec.get('off_hours_pct','N/A')}% of total
Weekday avg: {elec.get('avg_weekday_daily','N/A')} kWh/day | Weekend avg: {elec.get('avg_weekend_daily','N/A')} kWh/day
Month-on-month: {elec.get('month_on_month_pct','N/A')}%
Monthly breakdown: {json.dumps(elec.get('monthly_breakdown',{}))}
GAS: {gas.get('total_kwh','N/A')} kWh | £{gas.get('total_cost_gbp','N/A')}
{bms_section}
BENCHMARKS (CIBSE TM46 — Hotels):
- Good practice electricity: 195 kWh/m²/yr | Typical: 305 kWh/m²/yr
- Good practice fossil fuel: 285 kWh/m²/yr
UK rates: Electricity £0.28/kWh | Gas £0.07/kWh | Carbon: 0.207 kgCO₂/kWh

HOW TO RESPOND:
- Use your database tools for specific date/period questions
- When BMS data is available, actively correlate equipment parameters with energy patterns
- Flag any FAULT/ALARM ACTIVE entries prominently
- If setpoint temperatures look high vs best practice, flag as saving opportunity
- If equipment shows no overnight off, flag as potential waste
- Give practical, actionable recommendations with estimated savings
- If asked about ESOS, ISO 50001, procurement strategy, net zero — briefly acknowledge but explain Enterprise plan covers this
- Format: **bold** for key numbers, bullet points for lists"""


def build_senior_consultant_prompt(elec, gas, today, bms_context=""):
    bms_section = f"\n{bms_context}\n" if bms_context else "\nBMS DATA: No equipment parameters uploaded yet.\n"
    return f"""You are Effictra AI Senior Energy Consultant — a senior energy consultant with 20+ years of experience, provided exclusively on the Effictra AI Enterprise plan by Effictra Energy (effictraenergy.co.uk).

Today's date: {today}

YOUR FULL EXPERTISE:
- UK energy regulations: ESOS Phase 3, SECR, EPC, DEC, PPN 06/21, MEES
- International standards: ISO 50001, ISO 14001, ASHRAE 90.1, BREEAM, LEED, WELL
- All building types: hotels, offices, hospitals, retail, industrial, data centres, residential
- Technologies: HVAC, BMS/EMS, LED, heat pumps, solar PV, battery storage, CHP, EV charging
- Tariff structures: half-hourly metering, TOU, capacity charges, DUoS, TNUoS, BSUoS, CCL
- Carbon reporting: Scope 1/2/3, SECR narrative, TCFD, SBTi, net zero pathways, PAS 2060
- Benchmarking: CIBSE TM46, TM54, Carbon Trust, Display Energy Certificates, REEB
- BMS/controls optimisation: setpoint analysis, scheduling, sequence of operations review
- Demand side response, flexibility markets, FFR, DC, BM, smart grid participation
- Utility procurement: PPAs, flexible contracts, basket trading, green tariffs, REGOs
- Funding: SALIX loans, ECO4, UKRI, PSDS, HUG2, industrial energy transformation fund
- Investment appraisal: NPV, IRR, simple payback, lifecycle cost analysis
- Sustainability reporting: GRI, CDP, B Corp, ESG frameworks

THIS BUILDING'S DATA:
Electricity: {elec.get('total_kwh','N/A')} kWh total | £{elec.get('total_cost_gbp','N/A')} cost
Period: {elec.get('data_from','N/A')} to {elec.get('data_to','N/A')} ({elec.get('days_of_data','N/A')} days)
Avg daily: {elec.get('avg_daily_kwh','N/A')} kWh | Peak hour: {elec.get('peak_hour','N/A')}:00
Baseload: {elec.get('baseload_kwh','N/A')} kWh/h | Peak demand: {elec.get('peak_demand_kwh','N/A')} kWh/h
Off-hours (22:00-06:00): {elec.get('off_hours_pct','N/A')}% of total
Weekday avg: {elec.get('avg_weekday_daily','N/A')} kWh/day | Weekend avg: {elec.get('avg_weekend_daily','N/A')} kWh/day
Month-on-month: {elec.get('month_on_month_pct','N/A')}%
Monthly breakdown: {json.dumps(elec.get('monthly_breakdown',{}))}
GAS: {gas.get('total_kwh','N/A')} kWh | £{gas.get('total_cost_gbp','N/A')}
{bms_section}
BENCHMARKS (CIBSE TM46 — Hotels):
Good practice electricity: 195 kWh/m²/yr | Typical: 305 kWh/m²/yr
Good practice fossil fuel: 285 kWh/m²/yr | Typical: 420 kWh/m²/yr
UK rates: Electricity £0.28/kWh | Gas £0.07/kWh | Carbon intensity: 0.207 kgCO₂/kWh (DESNZ 2024)

HOW TO RESPOND:
- Use database tools for specific data questions
- When BMS data is available, provide detailed analysis: setpoint optimisation vs CIBSE guidance, scheduling from on/off patterns, fault diagnosis
- Flag active fault alarms immediately with root cause analysis and recommended actions
- Apply full consultancy expertise — compliance, strategy, procurement, funding
- Give specific recommendations with £/kWh savings, ROI, payback periods
- Reference relevant regulations, standards, funding schemes
- Be direct, confident and authoritative — you are the expert
- Format: **bold** for key numbers, bullet points for lists"""


# ─── AGENTIC LOOP ─────────────────────────────────────────────

async def run_analyst_chat(messages, system_prompt, org_id, conversation_id, original_messages):
    max_iterations=5; iteration=0; final_response=""
    while iteration < max_iterations:
        iteration += 1
        async with httpx.AsyncClient(timeout=60.0) as client:
            body={"model":ANTHROPIC_MODEL,"max_tokens":4000,"system":system_prompt,
                  "tools":ANALYST_TOOLS,"messages":messages}
            resp=await client.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
                json=body)
            resp.raise_for_status(); result=resp.json()
        stop_reason=result.get("stop_reason"); content=result.get("content",[])
        text_parts=[b["text"] for b in content if b["type"]=="text"]
        if text_parts: final_response="\n".join(text_parts)
        if stop_reason=="end_turn": break
        if stop_reason=="tool_use":
            tool_blocks=[b for b in content if b["type"]=="tool_use"]
            messages.append({"role":"assistant","content":content})
            tool_results=[]
            for tb in tool_blocks:
                print(f"[analyst] Tool: {tb['name']} — {tb['input']}")
                tool_result=execute_tool(tb["name"],tb["input"],org_id=org_id)
                tool_results.append({"type":"tool_result","tool_use_id":tb["id"],"content":tool_result})
            messages.append({"role":"user","content":tool_results}); continue
        break
    all_messages=[{"role":m["role"],"content":m["content"]} for m in original_messages]
    all_messages.append({"role":"assistant","content":final_response})
    if org_id:
        if conversation_id:
            supabase.table("ai_conversations").update(
                {"messages":all_messages,"updated_at":datetime.utcnow().isoformat()}
            ).eq("id",conversation_id).execute()
            return {"response":final_response,"conversation_id":conversation_id}
        else:
            title=original_messages[0]["content"][:60]+("..." if len(original_messages[0]["content"])>60 else "")
            result_db=supabase.table("ai_conversations").insert(
                {"org_id":org_id,"title":title,"messages":all_messages}).execute()
            if result_db.data:
                return {"response":final_response,"conversation_id":result_db.data[0]["id"]}
    return {"response":final_response,"conversation_id":conversation_id}

# ─── ROOT ─────────────────────────────────────────────────────

@app.get("/")
def root(): return {"message": "Effictra AI API running"}

@app.get("/health")
def health(): return {"status": "ok"}

# ─── AUTH ─────────────────────────────────────────────────────

@app.get("/auth/me")
def get_me(authorization: Optional[str]=Header(default=None)):
    auth_user,org=require_auth(authorization)
    tier=get_effective_tier(org)
    features=TIER_FEATURES.get(tier,{})
    return {"user":{"id":str(auth_user.id),"email":auth_user.email},"org":org,
            "tier":tier,"features":features,
            "trial_ends_at":org.get("trial_ends_at") if org else None}

@app.get("/tier")
def get_tier(org_id: Optional[str]=Query(default=None),
             authorization: Optional[str]=Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        try:
            auth_user,org=require_auth(authorization)
            tier=get_effective_tier(org)
            features=TIER_FEATURES.get(tier,{})
            return {"tier":tier,"features":features,
                    "trial_ends_at":org.get("trial_ends_at") if org else None,
                    "upgrade_url":f"{FRONTEND_URL}/pricing"}
        except HTTPException: pass
    tier=get_org_tier_by_id(org_id)
    features=TIER_FEATURES.get(tier,{})
    return {"tier":tier,"features":features,"upgrade_url":f"{FRONTEND_URL}/pricing"}

@app.get("/feature-flags")
def get_feature_flags(org_id: Optional[str]=Query(default=None),
                      authorization: Optional[str]=Header(default=None)):
    tier="basic"; org=None
    if authorization and authorization.startswith("Bearer "):
        try:
            _,org=require_auth(authorization)
            tier=get_effective_tier(org)
        except HTTPException:
            if org_id: tier=get_org_tier_by_id(org_id)
    elif org_id:
        tier=get_org_tier_by_id(org_id)
    features=TIER_FEATURES.get(tier, TIER_FEATURES["basic"])
    return {"tier":tier,"flags":features,"trial_ends_at":org.get("trial_ends_at") if org else None}

# ─── STRIPE ───────────────────────────────────────────────────

@app.post("/billing/create-checkout")
async def create_checkout_session(req: CheckoutRequest):
    if not STRIPE_SECRET_KEY: return {"success":False,"message":"Stripe not configured"}
    price_id=STRIPE_PRICE_MAP.get((req.plan,req.billing_period))
    if not price_id: return {"success":False,"message":f"No price for {req.plan}/{req.billing_period}"}
    org_result=supabase.table("organisations").select("*").eq("id",req.org_id).single().execute()
    if not org_result.data: return {"success":False,"message":"Organisation not found"}
    org=org_result.data
    success_url=req.success_url or f"{FRONTEND_URL}/dashboard?upgrade=success"
    cancel_url=req.cancel_url or f"{FRONTEND_URL}/pricing?upgrade=cancelled"
    try:
        async with httpx.AsyncClient() as client:
            customer_id=org.get("stripe_customer_id")
            if not customer_id:
                user_result=supabase.table("users").select("email,name").eq("org_id",req.org_id).limit(1).execute()
                email=user_result.data[0]["email"] if user_result.data else ""
                name=user_result.data[0]["name"] if user_result.data else ""
                cust_resp=await client.post("https://api.stripe.com/v1/customers",auth=(STRIPE_SECRET_KEY,""),
                    data={"email":email,"name":name,"metadata[org_id]":req.org_id})
                cust_resp.raise_for_status(); customer_id=cust_resp.json()["id"]
                supabase.table("organisations").update({"stripe_customer_id":customer_id}).eq("id",req.org_id).execute()
            resp=await client.post("https://api.stripe.com/v1/checkout/sessions",auth=(STRIPE_SECRET_KEY,""),
                data={"customer":customer_id,"mode":"subscription",
                      "line_items[0][price]":price_id,"line_items[0][quantity]":"1",
                      "success_url":success_url+"&session_id={CHECKOUT_SESSION_ID}",
                      "cancel_url":cancel_url,"metadata[org_id]":req.org_id,"metadata[plan]":req.plan,
                      "subscription_data[metadata][org_id]":req.org_id,
                      "subscription_data[metadata][plan]":req.plan,"allow_promotion_codes":"true"})
            resp.raise_for_status(); session=resp.json()
            return {"success":True,"checkout_url":session["url"],"session_id":session["id"]}
    except Exception as e: return {"success":False,"message":str(e)}

@app.post("/billing/create-portal")
async def create_billing_portal(org_id: str=Query(...)):
    if not STRIPE_SECRET_KEY: return {"success":False,"message":"Stripe not configured"}
    try:
        org=supabase.table("organisations").select("stripe_customer_id").eq("id",org_id).single().execute().data
        if not org or not org.get("stripe_customer_id"): return {"success":False,"message":"No Stripe customer"}
        async with httpx.AsyncClient() as client:
            resp=await client.post("https://api.stripe.com/v1/billing_portal/sessions",
                auth=(STRIPE_SECRET_KEY,""),
                data={"customer":org["stripe_customer_id"],"return_url":f"{FRONTEND_URL}/settings/billing"})
            resp.raise_for_status(); return {"success":True,"portal_url":resp.json()["url"]}
    except Exception as e: return {"success":False,"message":str(e)}

@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    payload=await request.body(); sig_header=request.headers.get("stripe-signature","")
    if STRIPE_WEBHOOK_SECRET:
        try:
            timestamp=sig_header.split("t=")[1].split(",")[0]
            signatures=[s.split("v1=")[1] for s in sig_header.split(",") if s.startswith("v1=")]
            signed_payload=f"{timestamp}.{payload.decode()}"
            expected=hmac.new(STRIPE_WEBHOOK_SECRET.encode(),signed_payload.encode(),hashlib.sha256).hexdigest()
            if expected not in signatures: raise HTTPException(status_code=400,detail="Invalid signature")
        except Exception: raise HTTPException(status_code=400,detail="Invalid signature")
    try:
        event=json.loads(payload); event_type=event.get("type"); print(f"[webhook] {event_type}")
        if event_type in ("checkout.session.completed","invoice.payment_succeeded"):
            obj=event["data"]["object"]; org_id=obj.get("metadata",{}).get("org_id"); plan=obj.get("metadata",{}).get("plan")
            if org_id and plan:
                supabase.table("organisations").update({"tier":plan,"updated_at":datetime.utcnow().isoformat()}).eq("id",org_id).execute()
                supabase.table("subscriptions").upsert({"org_id":org_id,
                    "stripe_subscription_id":obj.get("subscription") or obj.get("id"),
                    "plan":plan,"status":"active","updated_at":datetime.utcnow().isoformat()},
                    on_conflict="stripe_subscription_id").execute()
        elif event_type=="customer.subscription.deleted":
            obj=event["data"]["object"]; org_id=obj.get("metadata",{}).get("org_id")
            if org_id: supabase.table("organisations").update({"tier":"basic","updated_at":datetime.utcnow().isoformat()}).eq("id",org_id).execute()
        elif event_type=="invoice.payment_failed":
            obj=event["data"]["object"]; sub_id=obj.get("subscription")
            if sub_id: supabase.table("subscriptions").update({"status":"past_due","updated_at":datetime.utcnow().isoformat()}).eq("stripe_subscription_id",sub_id).execute()
        elif event_type=="customer.subscription.updated":
            obj=event["data"]["object"]; org_id=obj.get("metadata",{}).get("org_id"); plan=obj.get("metadata",{}).get("plan")
            if org_id and plan and obj.get("status")=="active":
                supabase.table("organisations").update({"tier":plan,"updated_at":datetime.utcnow().isoformat()}).eq("id",org_id).execute()
        return {"received":True}
    except Exception as e: print(f"[webhook] Error: {e}"); return {"received":True}

@app.get("/billing/subscription")
def get_subscription(org_id: str=Query(...)):
    try:
        result=supabase.table("subscriptions").select("*").eq("org_id",org_id).eq("status","active").order("created_at",desc=True).limit(1).execute()
        org=supabase.table("organisations").select("tier,trial_ends_at,stripe_customer_id").eq("id",org_id).single().execute().data
        return {"subscription":result.data[0] if result.data else None,"org":org}
    except Exception as e: return {"success":False,"message":str(e)}

@app.post("/admin/expire-trials")
def expire_trials():
    try: result=supabase.rpc("expire_trials").execute(); return {"success":True,"expired":result.data}
    except Exception as e: return {"success":False,"message":str(e)}

# ─── SITES ────────────────────────────────────────────────────

@app.get("/analytics/sites")
def get_sites():
    try:
        result=supabase.table("sites").select("id,name,lat,lng,timezone,base_temp,mode,address,building_type").eq("is_active",True).order("name").execute()
        return {"sites":result.data or []}
    except Exception as e: return {"success":False,"message":str(e)}

@app.post("/analytics/sites")
def create_site(site: SiteCreate):
    try:
        base_temp=auto_base_temp(site.lat,site.base_temp)
        result=supabase.table("sites").insert({"name":site.name,"lat":site.lat,"lng":site.lng,
            "timezone":site.timezone,"base_temp":base_temp,"mode":site.mode,
            "address":site.address,"building_type":site.building_type,"is_active":True}).execute()
        return {"success":True,"site":result.data[0] if result.data else None}
    except Exception as e: return {"success":False,"message":str(e)}

@app.put("/analytics/sites/{site_id}")
def update_site(site_id: str, site: SiteCreate):
    try:
        result=supabase.table("sites").update({"name":site.name,"lat":site.lat,"lng":site.lng,
            "timezone":site.timezone,"base_temp":site.base_temp,"mode":site.mode,
            "address":site.address,"building_type":site.building_type}).eq("id",site_id).execute()
        return {"success":True,"site":result.data[0] if result.data else None}
    except Exception as e: return {"success":False,"message":str(e)}

@app.delete("/analytics/sites/{site_id}")
def delete_site(site_id: str):
    try: supabase.table("sites").update({"is_active":False}).eq("id",site_id).execute(); return {"success":True,"deleted":site_id}
    except Exception as e: return {"success":False,"message":str(e)}

# ─── BMS — EQUIPMENT ──────────────────────────────────────────

@app.get("/bms/equipment")
def list_equipment(site_id: str=Query(...),
                   authorization: Optional[str]=Header(default=None),
                   org_id: Optional[str]=Query(default=None)):
    require_feature_jwt(authorization, org_id, "bms_parameters")
    try:
        result = (supabase.table("equipment")
                  .select("id,name,category,manufacturer,model,bms_ref,is_active,created_at")
                  .eq("site_id", site_id).eq("is_active", True)
                  .order("category").order("name").execute())
        return {"equipment": result.data or []}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.post("/bms/equipment")
def create_equipment(eq: EquipmentCreate,
                     authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, None, "bms_parameters")
    try:
        result = supabase.table("equipment").insert({
            "site_id": eq.site_id, "name": eq.name, "category": eq.category,
            "manufacturer": eq.manufacturer, "model": eq.model,
            "bms_ref": eq.bms_ref, "is_active": eq.is_active,
        }).execute()
        return {"success": True, "equipment": result.data[0] if result.data else None}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.put("/bms/equipment/{equipment_id}")
def update_equipment(equipment_id: str, eq: EquipmentUpdate,
                     authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, None, "bms_parameters")
    try:
        updates = {k: v for k, v in eq.dict().items() if v is not None}
        result = supabase.table("equipment").update(updates).eq("id", equipment_id).execute()
        return {"success": True, "equipment": result.data[0] if result.data else None}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.delete("/bms/equipment/{equipment_id}")
def delete_equipment(equipment_id: str,
                     authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, None, "bms_parameters")
    try:
        supabase.table("equipment").update({"is_active": False}).eq("id", equipment_id).execute()
        return {"success": True, "deleted": equipment_id}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

# ─── BMS — PARAMETERS ─────────────────────────────────────────

@app.get("/bms/equipment/{equipment_id}/parameters")
def list_parameters(equipment_id: str,
                    authorization: Optional[str]=Header(default=None),
                    org_id: Optional[str]=Query(default=None)):
    require_feature_jwt(authorization, org_id, "bms_parameters")
    try:
        result = (supabase.table("equipment_parameters")
                  .select("id,parameter_name,parameter_type,unit,bms_tag,created_at")
                  .eq("equipment_id", equipment_id).order("parameter_type").execute())
        return {"parameters": result.data or []}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.post("/bms/parameters")
def create_parameter(param: ParameterCreate,
                     authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, None, "bms_parameters")
    try:
        result = supabase.table("equipment_parameters").insert({
            "equipment_id": param.equipment_id, "parameter_name": param.parameter_name,
            "parameter_type": param.parameter_type, "unit": param.unit, "bms_tag": param.bms_tag,
        }).execute()
        return {"success": True, "parameter": result.data[0] if result.data else None}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.delete("/bms/parameters/{parameter_id}")
def delete_parameter(parameter_id: str,
                     authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, None, "bms_parameters")
    try:
        supabase.table("equipment_parameters").delete().eq("id", parameter_id).execute()
        return {"success": True, "deleted": parameter_id}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

# ─── BMS — READINGS (CSV UPLOAD) ──────────────────────────────

@app.post("/bms/parameters/{parameter_id}/upload")
async def upload_bms_readings(parameter_id: str,
                               file: UploadFile=File(...),
                               authorization: Optional[str]=Header(default=None)):
    """
    Upload BMS readings CSV for a specific parameter.
    Expected CSV: timestamp column + value column (+ optional text/status column).

    Examples:
      2024-01-15 08:00, 72.5          (temperature)
      2024-01-15 08:00, 1, ON         (on/off with text)
      2024-01-15 08:00, HEATING       (mode)
    """
    require_feature_jwt(authorization, None, "bms_parameters")
    try:
        param_check = (supabase.table("equipment_parameters")
                       .select("id,parameter_name,parameter_type")
                       .eq("id", parameter_id).single().execute())
        if not param_check.data:
            raise HTTPException(status_code=404, detail="Parameter not found")
        contents = await file.read()
        records = parse_bms_csv(contents, parameter_id=parameter_id)
        if not records:
            return {"success": False, "message": "No valid rows parsed from CSV"}
        inserted = 0
        for i in range(0, len(records), 500):
            supabase.table("equipment_readings").insert(records[i:i+500]).execute()
            inserted += len(records[i:i+500])
        print(f"[bms] Uploaded {inserted} readings for parameter {parameter_id}")
        return {
            "success": True, "parameter_id": parameter_id,
            "parameter_name": param_check.data["parameter_name"],
            "parameter_type": param_check.data["parameter_type"],
            "rows_inserted": inserted,
            "message": f"{inserted} readings uploaded successfully",
        }
    except HTTPException: raise
    except Exception as e:
        print(f"[bms] Upload error: {e}")
        return {"success": False, "message": str(e)}

@app.get("/bms/parameters/{parameter_id}/readings")
def get_readings(parameter_id: str,
                 start_date: Optional[str]=Query(default=None),
                 end_date: Optional[str]=Query(default=None),
                 limit: int=Query(default=500, le=5000),
                 authorization: Optional[str]=Header(default=None),
                 org_id: Optional[str]=Query(default=None)):
    require_feature_jwt(authorization, org_id, "bms_parameters")
    try:
        query = (supabase.table("equipment_readings")
                 .select("id,recorded_at,value,value_text")
                 .eq("parameter_id", parameter_id)
                 .order("recorded_at", desc=False))
        if start_date: query = query.gte("recorded_at", start_date)
        if end_date: query = query.lte("recorded_at", end_date + "T23:59:59")
        result = query.limit(limit).execute()
        return {"readings": result.data or [], "count": len(result.data or [])}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.delete("/bms/parameters/{parameter_id}/readings")
def delete_readings(parameter_id: str,
                    authorization: Optional[str]=Header(default=None)):
    """Delete all readings for a parameter (useful for re-upload)."""
    require_feature_jwt(authorization, None, "bms_parameters")
    try:
        supabase.table("equipment_readings").delete().eq("parameter_id", parameter_id).execute()
        return {"success": True, "message": f"All readings deleted for parameter {parameter_id}"}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.get("/bms/site/{site_id}/summary")
def get_site_bms_summary(site_id: str,
                          days: int=Query(default=7, ge=1, le=90),
                          authorization: Optional[str]=Header(default=None),
                          org_id: Optional[str]=Query(default=None)):
    """
    Summary of all BMS equipment and latest parameter values for a site.
    Used by the frontend equipment overview page.
    """
    require_feature_jwt(authorization, org_id, "bms_parameters")
    try:
        equipment_rows = (supabase.table("equipment")
                          .select("id,name,category,manufacturer,model,bms_ref")
                          .eq("site_id", site_id).eq("is_active", True)
                          .order("category").order("name").execute().data) or []
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        result = []
        for eq in equipment_rows:
            params = (supabase.table("equipment_parameters")
                      .select("id,parameter_name,parameter_type,unit")
                      .eq("equipment_id", eq["id"]).execute().data) or []
            param_summaries = []
            has_active_fault = False
            for param in params:
                latest = (supabase.table("equipment_readings")
                          .select("recorded_at,value,value_text")
                          .eq("parameter_id", param["id"])
                          .gte("recorded_at", cutoff)
                          .order("recorded_at", desc=True)
                          .limit(1).execute().data)
                latest_reading = latest[0] if latest else None
                fault_active = False
                if param["parameter_type"] == "fault_alarm" and latest_reading:
                    val = str(latest_reading.get("value","")).strip()
                    val_text = str(latest_reading.get("value_text","")).lower()
                    fault_active = val in ("1","1.0") or val_text in ("fault","alarm","true","active")
                    if fault_active: has_active_fault = True
                param_summaries.append({
                    "id": param["id"], "parameter_name": param["parameter_name"],
                    "parameter_type": param["parameter_type"], "unit": param.get("unit"),
                    "latest_value": latest_reading.get("value") if latest_reading else None,
                    "latest_value_text": latest_reading.get("value_text") if latest_reading else None,
                    "latest_recorded_at": latest_reading.get("recorded_at") if latest_reading else None,
                    "fault_active": fault_active,
                })
            result.append({
                "id": eq["id"], "name": eq["name"], "category": eq["category"],
                "manufacturer": eq.get("manufacturer"), "model": eq.get("model"),
                "bms_ref": eq.get("bms_ref"), "has_active_fault": has_active_fault,
                "parameters": param_summaries,
            })
        by_category = {}
        for item in result:
            cat = item["category"]
            if cat not in by_category: by_category[cat] = []
            by_category[cat].append(item)
        return {
            "site_id": site_id, "days": days,
            "equipment_count": len(result),
            "active_faults": sum(1 for item in result if item["has_active_fault"]),
            "equipment": result, "by_category": by_category,
        }
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

# ─── WEATHER NORMALISED ───────────────────────────────────────

@app.get("/analytics/weather-normalised")
async def get_weather_normalised(site_id: str=Query(...),org_id: Optional[str]=Query(default=None),
    start_date: Optional[str]=Query(default=None),end_date: Optional[str]=Query(default=None),
    authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization,org_id,"weather_normalisation")
    try:
        site=supabase.table("sites").select("*").eq("id",site_id).eq("is_active",True).single().execute().data
        if not site: return {"success":False,"message":"Site not found"}
        base_temp=site["base_temp"]; mode=resolve_mode(site["mode"],site["lat"],base_temp)
        today=datetime.utcnow().date()
        if not end_date: end_date=str(today-timedelta(days=1))
        if not start_date: start_date=str(today-timedelta(days=31))
        data=(supabase.table("energy_data").select("timestamp,consumption").gte("timestamp",start_date)
              .lte("timestamp",end_date+"T23:59:59").range(0,20000).execute().data)
        if not data: return {"success":False,"message":"No energy data"}
        df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
        df=df.dropna(subset=["timestamp"]); df["date"]=df["timestamp"].dt.date.astype(str)
        daily_df=df.groupby("date")["consumption"].sum().reset_index()
        daily_df.columns=["date","actual"]; daily_df=daily_df.sort_values("date").reset_index(drop=True)
        if daily_df.empty: return {"success":False,"message":"No daily data"}
        actual_start,actual_end=daily_df["date"].min(),daily_df["date"].max()
        dd=await fetch_degree_days(site["lat"],site["lng"],base_temp,actual_start,actual_end)
        rows=[{"date":row["date"],"actual":float(row["actual"]),
               "hdd":dd.get(row["date"],{}).get("hdd",0.0),"cdd":dd.get(row["date"],{}).get("cdd",0.0),
               "mean_temp":dd.get(row["date"],{}).get("mean_temp")} for _,row in daily_df.iterrows()]
        bh,bc,bl=estimate_sensitivity([r["actual"] for r in rows],[r["hdd"] for r in rows],[r["cdd"] for r in rows],mode)
        results=[{"date":r["date"],"actual":round(r["actual"],1),
                  "normalised":round(r["actual"]-(round((r["hdd"]*bh)+(r["cdd"]*bc),1)),1),
                  "weatherImpact":round((r["hdd"]*bh)+(r["cdd"]*bc),1),
                  "hdd":r["hdd"],"cdd":r["cdd"],"meanTemp":r["mean_temp"]} for r in rows]
        ta=round(sum(r["actual"] for r in results),1)
        tn=round(sum(r["normalised"] for r in results),1)
        tw=round(sum(r["weatherImpact"] for r in results),1)
        return {"site":{"id":site["id"],"name":site["name"],"baseTemp":base_temp,"mode":mode},
                "dateRange":{"start":actual_start,"end":actual_end},
                "coefficients":{"beta_h":round(bh,3),"beta_c":round(bc,3),"baseload":round(bl,1)},
                "summary":{"totalActual":ta,"totalNormalised":tn,"totalWeatherImpact":tw,
                           "weatherImpactPct":round(tw/ta*100,1) if ta else 0},"daily":results}
    except Exception as e: return {"success":False,"message":str(e)}

# ─── ANOMALIES ────────────────────────────────────────────────

# ─── ANOMALY DETECTION ENGINE ─────────────────────────────────
# This section replaces the simple in-memory anomaly detection
# with a full 3-layer engine: statistical, BMS-correlated, trend

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alerts@effictraenergy.co.uk")

# ── BASELINE MANAGEMENT ───────────────────────────────────────

def recalculate_baselines(org_id: str, energy_type: str = "electricity", days: int = 90):
    """
    Build site-specific hourly baselines from last N days of data.
    Stores median + std_dev per (day_of_week, hour_of_day).
    """
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        if energy_type == "electricity":
            q = (supabase.table("energy_data").select("timestamp,consumption")
                 .eq("org_id", org_id).gte("timestamp", cutoff).range(0, 50000))
        else:
            q = (supabase.table("gas_data").select("timestamp,consumption")
                 .eq("org_id", org_id).gte("timestamp", cutoff).range(0, 50000))

        data = q.execute().data or []
        if not data:
            return {"success": False, "message": f"No {energy_type} data found"}

        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        df["dow"] = df["timestamp"].dt.dayofweek
        df["hour"] = df["timestamp"].dt.hour

        # Calculate baseline per (dow, hour)
        baselines = []
        for (dow, hour), group in df.groupby(["dow", "hour"]):
            vals = group["consumption"].values
            if len(vals) < 3:
                continue
            baselines.append({
                "org_id": org_id,
                "energy_type": energy_type,
                "day_of_week": int(dow),
                "hour_of_day": int(hour),
                "baseline_kwh": round(float(np.median(vals)), 4),
                "std_dev": round(float(np.std(vals)), 4),
                "sample_count": int(len(vals)),
                "calculated_at": datetime.utcnow().isoformat(),
            })

        # Upsert baselines
        for b in baselines:
            supabase.table("energy_baselines").upsert(
                b, on_conflict="org_id,energy_type,day_of_week,hour_of_day"
            ).execute()

        print(f"[baselines] {org_id} {energy_type}: {len(baselines)} baselines calculated")
        return {"success": True, "baselines_calculated": len(baselines)}
    except Exception as e:
        print(f"[baselines] Error: {e}")
        return {"success": False, "message": str(e)}


def get_baselines(org_id: str, energy_type: str) -> dict:
    """Load baselines as {(dow, hour): (baseline_kwh, std_dev)} dict."""
    try:
        result = (supabase.table("energy_baselines")
                  .select("day_of_week,hour_of_day,baseline_kwh,std_dev")
                  .eq("org_id", org_id)
                  .eq("energy_type", energy_type)
                  .execute())
        return {
            (row["day_of_week"], row["hour_of_day"]): (row["baseline_kwh"], row["std_dev"])
            for row in (result.data or [])
        }
    except Exception:
        return {}


# ── BMS CORRELATION ───────────────────────────────────────────

def correlate_with_bms(timestamp_str: str, window_hours: int = 2) -> list:
    """
    For a given timestamp, find any BMS anomalies within ±window_hours.
    Returns list of correlation objects.
    """
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", ""))
        window_start = (ts - timedelta(hours=window_hours)).isoformat()
        window_end = (ts + timedelta(hours=window_hours)).isoformat()
        correlations = []

        # Get all equipment
        equipment_rows = (supabase.table("equipment")
                          .select("id,name,category")
                          .eq("is_active", True).execute().data) or []

        for eq in equipment_rows:
            params = (supabase.table("equipment_parameters")
                      .select("id,parameter_name,parameter_type,unit")
                      .eq("equipment_id", eq["id"]).execute().data) or []

            for param in params:
                readings = (supabase.table("equipment_readings")
                            .select("recorded_at,value,value_text")
                            .eq("parameter_id", param["id"])
                            .gte("recorded_at", window_start)
                            .lte("recorded_at", window_end)
                            .order("recorded_at", desc=True)
                            .limit(10).execute().data) or []

                if not readings:
                    continue

                ptype = param["parameter_type"]

                # Fault alarm active
                if ptype == "fault_alarm":
                    active = [r for r in readings if
                              str(r.get("value", "")).strip() in ("1", "1.0") or
                              str(r.get("value_text", "")).lower() in ("fault", "alarm", "true", "active")]
                    if active:
                        correlations.append({
                            "equipment": eq["name"],
                            "category": eq["category"],
                            "parameter": param["parameter_name"],
                            "type": "fault_alarm",
                            "value": "ACTIVE",
                            "recorded_at": active[0]["recorded_at"],
                            "description": f"{eq['name']} fault alarm active during energy anomaly"
                        })

                # Flow/return temp outlier
                elif ptype in ("flow_temp", "return_temp"):
                    vals = [float(r["value"]) for r in readings if r.get("value") is not None]
                    if vals:
                        avg_temp = sum(vals) / len(vals)
                        unit = param.get("unit") or "°C"
                        if ptype == "flow_temp" and avg_temp > 80:
                            correlations.append({
                                "equipment": eq["name"],
                                "category": eq["category"],
                                "parameter": param["parameter_name"],
                                "type": "temp_high",
                                "value": f"{round(avg_temp, 1)}{unit}",
                                "recorded_at": readings[0]["recorded_at"],
                                "description": f"{eq['name']} flow temp elevated ({round(avg_temp,1)}{unit}) — possible overfiring"
                            })
                        elif ptype == "return_temp" and avg_temp > 70:
                            correlations.append({
                                "equipment": eq["name"],
                                "category": eq["category"],
                                "parameter": param["parameter_name"],
                                "type": "temp_high",
                                "value": f"{round(avg_temp, 1)}{unit}",
                                "recorded_at": readings[0]["recorded_at"],
                                "description": f"{eq['name']} return temp high ({round(avg_temp,1)}{unit})"
                            })

                # Heating valve stuck open
                elif ptype == "Htg_Vlv_pos":
                    vals = [float(r["value"]) for r in readings if r.get("value") is not None]
                    if vals and max(vals) > 90:
                        correlations.append({
                            "equipment": eq["name"],
                            "category": eq["category"],
                            "parameter": param["parameter_name"],
                            "type": "valve_stuck",
                            "value": f"{round(max(vals), 1)}%",
                            "recorded_at": readings[0]["recorded_at"],
                            "description": f"{eq['name']} heating valve stuck open ({round(max(vals),1)}%)"
                        })

                # On/Off — equipment running at unusual time
                elif ptype == "on_off":
                    ts_hour = ts.hour
                    on_readings = [r for r in readings if
                                   str(r.get("value", "")).strip() in ("1", "1.0") or
                                   str(r.get("value_text", "")).lower() in ("on", "true")]
                    if on_readings and (ts_hour < 6 or ts_hour >= 22):
                        correlations.append({
                            "equipment": eq["name"],
                            "category": eq["category"],
                            "parameter": param["parameter_name"],
                            "type": "off_hours_running",
                            "value": "ON",
                            "recorded_at": on_readings[0]["recorded_at"],
                            "description": f"{eq['name']} running at {ts_hour:02d}:00 (off-hours)"
                        })

        return correlations
    except Exception as e:
        print(f"[bms_correlation] Error: {e}")
        return []


# ── STATISTICAL ANOMALY DETECTION ─────────────────────────────

def detect_statistical_anomalies(
    org_id: str,
    energy_type: str = "electricity",
    days: int = 90,
    use_stored_baselines: bool = True
) -> list:
    """
    Detect statistical anomalies in electricity or gas data.
    Uses stored baselines if available, otherwise calculates on-the-fly.
    Returns list of anomaly dicts ready for DB insert.
    Hard-limited to last 90 days regardless of input.
    """
    try:
        days = min(days, 90)  # Hard cap — never scan more than 3 months
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        table = "energy_data" if energy_type == "electricity" else "gas_data"
        q = (supabase.table(table).select("timestamp,consumption")
             .eq("org_id", org_id).gte("timestamp", cutoff).range(0, 50000))
        data = q.execute().data or []
        if not data:
            return []

        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["dow"] = df["timestamp"].dt.dayofweek
        df["hour"] = df["timestamp"].dt.hour
        df["date"] = df["timestamp"].dt.date.astype(str)
        df["is_weekend"] = df["dow"] >= 5
        df["is_off_hours"] = (df["hour"] < 6) | (df["hour"] >= 22)

        # Load or build baselines
        baselines = get_baselines(org_id, energy_type) if use_stored_baselines else {}
        if not baselines:
            # Build in-memory baseline
            for (dow, hour), group in df.groupby(["dow", "hour"]):
                vals = group["consumption"].values
                if len(vals) >= 3:
                    baselines[(int(dow), int(hour))] = (float(np.median(vals)), float(np.std(vals)))

        anomalies = []
        for _, row in df.iterrows():
            actual = float(row["consumption"])
            dow = int(row["dow"])
            hour = int(row["hour"])
            key = (dow, hour)

            if key not in baselines:
                continue

            baseline_kwh, std_dev = baselines[key]
            if baseline_kwh <= 0:
                continue

            dev_pct = ((actual - baseline_kwh) / baseline_kwh) * 100
            std_devs = (actual - baseline_kwh) / std_dev if std_dev > 0 else 0
            abs_std = abs(std_devs)
            abs_pct = abs(dev_pct)

            # Detection thresholds
            is_anomaly = (
                (abs_std >= 3.0 and abs_pct >= 20) or
                (abs_std >= 2.0 and abs_pct >= 30) or
                (abs_std >= 1.5 and abs_pct >= 50)
            )

            if not is_anomaly:
                continue

            # Determine type and severity
            if actual > baseline_kwh:
                a_type = "off_hours_spike" if row["is_off_hours"] else \
                         "weekend_spike" if row["is_weekend"] else "spike"
            else:
                a_type = "drop"

            sev = ("high" if (abs_std >= 3.0 or abs_pct >= 100) else
                   "medium" if (abs_std >= 2.0 or abs_pct >= 50) else "low")

            ts_str = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S")

            anomalies.append({
                "org_id": org_id,
                "anomaly_type": a_type,
                "energy_type": energy_type,
                "severity": sev,
                "timestamp_start": ts_str,
                "timestamp_end": ts_str,
                "actual_value": round(actual, 4),
                "expected_value": round(baseline_kwh, 4),
                "deviation_pct": round(dev_pct, 2),
                "std_deviations": round(std_devs, 2),
                "description": (
                    f"{energy_type.capitalize()} {'spike' if actual > baseline_kwh else 'drop'}: "
                    f"{round(actual, 2)} kWh vs expected {round(baseline_kwh, 2)} kWh "
                    f"({dev_pct:+.1f}%, {std_devs:.1f}σ) at "
                    f"{row['timestamp'].strftime('%a %d %b %Y %H:%M')}"
                ),
                "bms_correlation": [],
                "trend_data": {},
                "detected_at": datetime.utcnow().isoformat(),
            })

        print(f"[anomaly] {org_id} {energy_type}: {len(anomalies)} statistical anomalies found")
        return anomalies

    except Exception as e:
        print(f"[anomaly] Statistical detection error: {e}")
        import traceback; traceback.print_exc()
        return []


# ── TREND ANOMALY DETECTION ────────────────────────────────────

def detect_trend_anomalies(org_id: str) -> list:
    """
    Detect trend anomalies: baseload creep, MoM increases, peak demand trend.
    Uses last 90 days of data.
    """
    anomalies = []
    try:
        cutoff = (datetime.utcnow() - timedelta(days=90)).isoformat()
        data = (supabase.table("energy_data").select("timestamp,consumption")
                .eq("org_id", org_id).gte("timestamp", cutoff)
                .range(0, 50000).execute().data) or []
        if not data:
            return []

        df = pd.DataFrame(data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["month"] = df["timestamp"].dt.month
        now = datetime.utcnow()

        # 1. Baseload creep — overnight avg this week vs 4 weeks ago
        off_hours = df[df["hour"].isin([0, 1, 2, 3, 4, 5])]
        if not off_hours.empty:
            recent_cutoff = now - timedelta(days=7)
            old_cutoff = now - timedelta(days=28)
            recent_bl = off_hours[off_hours["timestamp"] >= recent_cutoff]["consumption"].mean()
            old_bl = off_hours[(off_hours["timestamp"] >= old_cutoff) &
                               (off_hours["timestamp"] < recent_cutoff)]["consumption"].mean()
            if not np.isnan(recent_bl) and not np.isnan(old_bl) and old_bl > 0:
                bl_change = ((recent_bl - old_bl) / old_bl) * 100
                if bl_change > 15:
                    anomalies.append({
                        "org_id": org_id,
                        "anomaly_type": "trend_baseload_creep",
                        "energy_type": "electricity",
                        "severity": "high" if bl_change > 30 else "medium",
                        "timestamp_start": recent_cutoff.isoformat(),
                        "timestamp_end": now.isoformat(),
                        "actual_value": round(float(recent_bl), 2),
                        "expected_value": round(float(old_bl), 2),
                        "deviation_pct": round(bl_change, 1),
                        "std_deviations": 0,
                        "description": (
                            f"Overnight baseload has increased {bl_change:.1f}% over the past 4 weeks "
                            f"({round(old_bl,2)} → {round(recent_bl,2)} kWh/h avg). "
                            f"Possible causes: new equipment added, equipment not turning off, meter issue."
                        ),
                        "bms_correlation": [],
                        "trend_data": {"old_baseline": round(float(old_bl), 2),
                                       "new_baseline": round(float(recent_bl), 2),
                                       "change_pct": round(bl_change, 1)},
                        "detected_at": now.isoformat(),
                    })

        # 2. Month-on-month increase
        daily = df.groupby("date")["consumption"].sum().reset_index()
        daily["month"] = pd.to_datetime(daily["date"]).dt.month
        monthly = daily.groupby("month")["consumption"].sum()
        if len(monthly) >= 2:
            months_sorted = monthly.sort_index()
            last_m = float(months_sorted.iloc[-1])
            prev_m = float(months_sorted.iloc[-2])
            if prev_m > 0:
                mom_change = ((last_m - prev_m) / prev_m) * 100
                if mom_change > 20:
                    anomalies.append({
                        "org_id": org_id,
                        "anomaly_type": "trend_mom_increase",
                        "energy_type": "electricity",
                        "severity": "high" if mom_change > 40 else "medium",
                        "timestamp_start": (now - timedelta(days=30)).isoformat(),
                        "timestamp_end": now.isoformat(),
                        "actual_value": round(last_m, 1),
                        "expected_value": round(prev_m, 1),
                        "deviation_pct": round(mom_change, 1),
                        "std_deviations": 0,
                        "description": (
                            f"Month-on-month electricity increased {mom_change:.1f}% "
                            f"({round(prev_m,1)} → {round(last_m,1)} kWh). "
                            f"This exceeds typical seasonal variation (>20%)."
                        ),
                        "bms_correlation": [],
                        "trend_data": {"prev_month_kwh": round(prev_m, 1),
                                       "curr_month_kwh": round(last_m, 1),
                                       "change_pct": round(mom_change, 1)},
                        "detected_at": now.isoformat(),
                    })

        # 3. Peak demand trend — rolling 4-week peaks
        weekly_peak = df.groupby("week")["consumption"].max()
        if len(weekly_peak) >= 4:
            recent_4w = weekly_peak.tail(4).values
            older_4w = weekly_peak.iloc[-8:-4].values if len(weekly_peak) >= 8 else None
            if older_4w is not None and len(older_4w) >= 4:
                recent_avg_peak = float(np.mean(recent_4w))
                older_avg_peak = float(np.mean(older_4w))
                if older_avg_peak > 0:
                    peak_change = ((recent_avg_peak - older_avg_peak) / older_avg_peak) * 100
                    if peak_change > 15:
                        anomalies.append({
                            "org_id": org_id,
                            "anomaly_type": "trend_peak_demand",
                            "energy_type": "electricity",
                            "severity": "high" if peak_change > 30 else "medium",
                            "timestamp_start": (now - timedelta(days=28)).isoformat(),
                            "timestamp_end": now.isoformat(),
                            "actual_value": round(recent_avg_peak, 2),
                            "expected_value": round(older_avg_peak, 2),
                            "deviation_pct": round(peak_change, 1),
                            "std_deviations": 0,
                            "description": (
                                f"Peak demand trending up {peak_change:.1f}% over last 4 weeks "
                                f"(avg peak {round(older_avg_peak,2)} → {round(recent_avg_peak,2)} kWh). "
                                f"Review capacity charges and peak-shifting opportunities."
                            ),
                            "bms_correlation": [],
                            "trend_data": {"older_avg_peak": round(older_avg_peak, 2),
                                           "recent_avg_peak": round(recent_avg_peak, 2),
                                           "change_pct": round(peak_change, 1)},
                            "detected_at": now.isoformat(),
                        })

    except Exception as e:
        print(f"[anomaly] Trend detection error: {e}")
        import traceback; traceback.print_exc()

    print(f"[anomaly] {org_id}: {len(anomalies)} trend anomalies found")
    return anomalies


# ── EMAIL ALERTS ──────────────────────────────────────────────

async def send_anomaly_alert(org_id: str, anomaly: dict):
    """Send email alert for high-severity anomaly via SendGrid."""
    try:
        settings = (supabase.table("alert_settings")
                    .select("*").eq("org_id", org_id).single().execute().data)
        if not settings or not settings.get("email_alerts"):
            return
        if settings.get("min_severity") == "high" and anomaly.get("severity") != "high":
            return

        to_email = settings.get("alert_email")
        if not to_email:
            return

        bms = anomaly.get("bms_correlation", [])
        bms_text = ""
        if bms:
            bms_text = "\n\nBMS Correlation:\n" + "\n".join(
                f"  • {c['description']}" for c in bms
            )

        subject = f"⚠️ {anomaly['severity'].upper()} Energy Anomaly Detected"
        body = f"""Effictra AI has detected a {anomaly['severity']} severity energy anomaly.

Type: {anomaly['anomaly_type'].replace('_', ' ').title()}
Energy: {anomaly['energy_type'].capitalize()}
Time: {anomaly['timestamp_start']}
Actual: {anomaly.get('actual_value', 'N/A')} kWh
Expected: {anomaly.get('expected_value', 'N/A')} kWh
Deviation: {anomaly.get('deviation_pct', 0):+.1f}%

Details: {anomaly.get('description', '')}{bms_text}

View in dashboard: {FRONTEND_URL}/anomalies

---
Effictra AI Energy Intelligence Platform
To manage alert settings, visit {FRONTEND_URL}/settings
"""

        if SENDGRID_API_KEY:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    headers={"Authorization": f"Bearer {SENDGRID_API_KEY}",
                             "Content-Type": "application/json"},
                    json={
                        "personalizations": [{"to": [{"email": to_email}]}],
                        "from": {"email": ALERT_FROM_EMAIL, "name": "Effictra AI"},
                        "subject": subject,
                        "content": [{"type": "text/plain", "value": body}]
                    }
                )
                if resp.status_code == 202:
                    supabase.table("anomalies").update({
                        "alert_sent": True,
                        "alert_sent_at": datetime.utcnow().isoformat()
                    }).eq("id", anomaly["id"]).execute()
                    print(f"[alert] Sent to {to_email} for anomaly {anomaly['id']}")
                else:
                    print(f"[alert] SendGrid error {resp.status_code}: {resp.text}")

    except Exception as e:
        print(f"[alert] Error sending alert: {e}")


# ── FULL DETECTION RUN ────────────────────────────────────────

async def run_full_anomaly_detection(org_id: str, days: int = 90, enrich_bms: bool = True):
    """
    Run all three layers of anomaly detection for an org.
    Saves results to anomalies table and sends alerts.
    """
    days = min(days, 90)  # Hard cap — never scan more than 3 months
    print(f"[anomaly] Starting full detection for org={org_id} days={days} (max 90)")
    saved = 0
    alerts_sent = 0

    # Layer 1: Statistical — electricity
    elec_anomalies = detect_statistical_anomalies(org_id, "electricity", days)
    # Layer 1: Statistical — gas
    gas_anomalies = detect_statistical_anomalies(org_id, "gas", days)
    # Layer 3: Trend
    trend_anomalies = detect_trend_anomalies(org_id)

    all_anomalies = elec_anomalies + gas_anomalies + trend_anomalies

    # Layer 2: BMS correlation enrichment
    if enrich_bms:
        for a in all_anomalies:
            if a["anomaly_type"] not in ("trend_baseload_creep", "trend_mom_increase", "trend_peak_demand"):
                correlations = correlate_with_bms(a["timestamp_start"])
                a["bms_correlation"] = correlations

    # Deduplicate — avoid inserting anomalies we already have
    # Check last 24h for existing records
    cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    existing = (supabase.table("anomalies")
                .select("timestamp_start,anomaly_type,energy_type")
                .eq("org_id", org_id)
                .gte("detected_at", cutoff_24h)
                .execute().data) or []
    existing_set = {(r["timestamp_start"][:16], r["anomaly_type"], r["energy_type"])
                    for r in existing}

    # Save new anomalies
    to_save = []
    for a in all_anomalies:
        key = (a["timestamp_start"][:16], a["anomaly_type"], a["energy_type"])
        if key not in existing_set:
            to_save.append(a)

    for i in range(0, len(to_save), 100):
        batch = to_save[i:i+100]
        result = supabase.table("anomalies").insert(batch).execute()
        if result.data:
            saved += len(result.data)
            # Send alerts for high severity
            for saved_row in result.data:
                if saved_row.get("severity") == "high":
                    await send_anomaly_alert(org_id, saved_row)
                    alerts_sent += 1

    print(f"[anomaly] Done: {saved} new anomalies saved, {alerts_sent} alerts sent")
    return {"saved": saved, "alerts_sent": alerts_sent, "total_detected": len(all_anomalies)}


# ── ANOMALY ENDPOINTS ─────────────────────────────────────────

@app.get("/anomalies")
def get_anomalies(
    days: int = Query(default=90, ge=7, le=90),
    severity: Optional[str] = Query(default=None),
    anomaly_type: Optional[str] = Query(default=None),
    energy_type: Optional[str] = Query(default=None),
    org_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass

    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Fetch from anomalies table
        q = (supabase.table("anomalies").select("*")
             .gte("timestamp_start", cutoff)
             .order("timestamp_start", desc=True)
             .limit(1000))
        if resolved_org_id: q = q.eq("org_id", resolved_org_id)
        if severity: q = q.eq("severity", severity.lower())
        if anomaly_type: q = q.eq("anomaly_type", anomaly_type.lower())
        if energy_type: q = q.eq("energy_type", energy_type.lower())

        anomalies = q.execute().data or []

        # Build summary
        sev_order = {"high": 0, "medium": 1, "low": 2}
        summary = {
            "total": len(anomalies),
            "high": sum(1 for a in anomalies if a["severity"] == "high"),
            "medium": sum(1 for a in anomalies if a["severity"] == "medium"),
            "low": sum(1 for a in anomalies if a["severity"] == "low"),
            "spikes": sum(1 for a in anomalies if "spike" in a["anomaly_type"]),
            "drops": sum(1 for a in anomalies if a["anomaly_type"] == "drop"),
            "trends": sum(1 for a in anomalies if a["anomaly_type"].startswith("trend_")),
            "bms_correlated": sum(1 for a in anomalies if a.get("bms_correlation")),
            "unacknowledged": sum(1 for a in anomalies if not a.get("acknowledged")),
        }

        # Chart data — daily consumption with anomaly markers
        if resolved_org_id:
            elec_q = (supabase.table("energy_data").select("timestamp,consumption")
                      .eq("org_id", resolved_org_id)
                      .gte("timestamp", cutoff).range(0, 20000))
            elec_data = elec_q.execute().data or []
        else:
            elec_data = []

        chart_data = []
        avg_daily = 0
        heatmap = [[0]*24 for _ in range(7)]

        if elec_data:
            df = pd.DataFrame(elec_data)
            df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
            df = df.dropna(subset=["consumption"])
            df = parse_timestamps_naive(df)
            df["date"] = df["timestamp"].dt.date.astype(str)
            daily = df.groupby("date")["consumption"].sum().reset_index()

            anomaly_dates = {}
            for a in anomalies:
                d = a["timestamp_start"][:10]
                if d not in anomaly_dates or sev_order.get(a["severity"], 2) < sev_order.get(anomaly_dates[d]["sev"], 2):
                    anomaly_dates[d] = {"sev": a["severity"]}

            chart_data = [{
                "date": str(row["date"]),
                "consumption": round(float(row["consumption"]), 2),
                "hasAnomaly": str(row["date"]) in anomaly_dates,
                "anomalySev": anomaly_dates[str(row["date"])]["sev"] if str(row["date"]) in anomaly_dates else None,
                "anomalyCount": sum(1 for a in anomalies if a["timestamp_start"][:10] == str(row["date"]))
            } for _, row in daily.iterrows()]

            avg_daily = round(float(df.groupby("date")["consumption"].sum().mean()), 2)

            for _, row in df.iterrows():
                d, h = row["timestamp"].dayofweek, row["timestamp"].hour
                heatmap[d][h] += 1

        # Format anomalies for frontend
        formatted = []
        for a in anomalies:
            formatted.append({
                "id": a["id"],
                "timestamp": a["timestamp_start"],
                "date": a["timestamp_start"][:10],
                "hour": int(a["timestamp_start"][11:13]) if len(a["timestamp_start"]) > 11 else 0,
                "hourLabel": f"{a['timestamp_start'][11:13]}:00" if len(a["timestamp_start"]) > 11 else "00:00",
                "anomalyType": a["anomaly_type"],
                "energyType": a["energy_type"],
                "actual": a.get("actual_value"),
                "expected": a.get("expected_value"),
                "deviationPct": a.get("deviation_pct"),
                "stdDeviations": a.get("std_deviations"),
                "severity": a["severity"],
                "type": "spike" if "spike" in a["anomaly_type"] else
                        "drop" if a["anomaly_type"] == "drop" else
                        "trend" if a["anomaly_type"].startswith("trend_") else "other",
                "description": a.get("description", ""),
                "bmsCorrelation": a.get("bms_correlation", []),
                "acknowledged": a.get("acknowledged", False),
                "dowLabel": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][
                    datetime.fromisoformat(a["timestamp_start"]).weekday()
                ] if a.get("timestamp_start") else "",
            })

        return {
            "anomalies": formatted,
            "summary": summary,
            "chartData": chart_data,
            "avgDaily": avg_daily,
            "heatmap": heatmap,
            "totalScanned": len(elec_data),
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"success": False, "message": str(e)}


@app.get("/anomalies/{anomaly_id}")
def get_anomaly_detail(anomaly_id: str):
    """Get single anomaly with full BMS correlation detail."""
    try:
        result = supabase.table("anomalies").select("*").eq("id", anomaly_id).single().execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        return {"anomaly": result.data}
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}


@app.post("/anomalies/{anomaly_id}/acknowledge")
def acknowledge_anomaly(anomaly_id: str, authorization: Optional[str] = Header(default=None)):
    """Mark anomaly as acknowledged."""
    try:
        auth_user = get_user_from_token(authorization)
        supabase.table("anomalies").update({
            "acknowledged": True,
            "acknowledged_at": datetime.utcnow().isoformat(),
            "acknowledged_by": str(auth_user.id) if auth_user else None,
        }).eq("id", anomaly_id).execute()
        return {"success": True}
    except Exception as e: return {"success": False, "message": str(e)}


@app.post("/anomalies/detect")
async def trigger_anomaly_detection(
    background_tasks: BackgroundTasks,
    org_id: Optional[str] = Query(default=None),
    days: int = Query(default=90, ge=7, le=90),
    authorization: Optional[str] = Header(default=None)
):
    """Trigger a fresh anomaly detection run. Always scans last 90 days max."""
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    if not resolved_org_id:
        raise HTTPException(status_code=400, detail="org_id required")
    background_tasks.add_task(run_full_anomaly_detection, org_id=resolved_org_id, days=90)
    return {"success": True, "message": "Anomaly detection started for last 90 days (3 months)"}


@app.get("/anomalies/trends/summary")
def get_trend_anomalies(
    org_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    """Get trend anomalies only."""
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    try:
        q = (supabase.table("anomalies").select("*")
             .like("anomaly_type", "trend_%")
             .order("detected_at", desc=True).limit(50))
        if resolved_org_id: q = q.eq("org_id", resolved_org_id)
        return {"trends": q.execute().data or []}
    except Exception as e: return {"success": False, "message": str(e)}


@app.post("/baselines/recalculate")
async def recalculate_all_baselines(
    background_tasks: BackgroundTasks,
    org_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    """Rebuild energy baselines from last 90 days."""
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    if not resolved_org_id:
        raise HTTPException(status_code=400, detail="org_id required")

    def rebuild():
        recalculate_baselines(resolved_org_id, "electricity")
        recalculate_baselines(resolved_org_id, "gas")

    background_tasks.add_task(rebuild)
    return {"success": True, "message": "Baseline recalculation started"}


@app.get("/baselines")
def get_baselines_endpoint(
    org_id: Optional[str] = Query(default=None),
    energy_type: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    """View stored baselines for an org."""
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    try:
        q = supabase.table("energy_baselines").select("*")
        if resolved_org_id: q = q.eq("org_id", resolved_org_id)
        if energy_type: q = q.eq("energy_type", energy_type)
        result = q.order("energy_type").order("day_of_week").order("hour_of_day").execute()
        return {"baselines": result.data or [], "count": len(result.data or [])}
    except Exception as e: return {"success": False, "message": str(e)}


@app.get("/alert-settings")
def get_alert_settings(
    org_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    try:
        result = (supabase.table("alert_settings").select("*")
                  .eq("org_id", resolved_org_id).single().execute())
        if not result.data:
            return {"settings": {"email_alerts": True, "alert_email": None, "min_severity": "high"}}
        return {"settings": result.data}
    except Exception as e:
        return {"settings": {"email_alerts": True, "alert_email": None, "min_severity": "high"}}


@app.put("/alert-settings")
def update_alert_settings(
    settings: dict,
    org_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    try:
        payload = {
            "org_id": resolved_org_id,
            "email_alerts": settings.get("email_alerts", True),
            "alert_email": settings.get("alert_email"),
            "min_severity": settings.get("min_severity", "high"),
            "updated_at": datetime.utcnow().isoformat(),
        }
        supabase.table("alert_settings").upsert(payload, on_conflict="org_id").execute()
        return {"success": True}
    except Exception as e: return {"success": False, "message": str(e)}



# ─── AI INSIGHTS ──────────────────────────────────────────────

# Period types and their rules
# all_time   → once ever (never regenerate)
# last_year  → once ever
# last_3_months → once per month (regenerate next month)
# last_1_month  → once per month (regenerate next month)

PERIOD_LABELS = {
    "all_time":       "All available data",
    "last_year":      "Last 12 months",
    "last_3_months":  "Last 3 months",
    "last_1_month":   "Last month",
}

def get_period_date_range(period_type: str):
    """Return (start_date, end_date) strings for a period type."""
    today = datetime.utcnow().date()
    end = str(today)
    if period_type == "all_time":
        return None, end  # No start filter
    elif period_type == "last_year":
        return str(today - timedelta(days=365)), end
    elif period_type == "last_3_months":
        return str(today - timedelta(days=90)), end
    elif period_type == "last_1_month":
        return str(today - timedelta(days=30)), end
    return None, end

def check_generation_allowed(org_id: str, period_type: str):
    """
    Check if generation is allowed for this org + period.
    Returns (allowed: bool, reason: str, existing_id: str|None)
    """
    try:
        result = (supabase.table("ai_insights")
                  .select("id,generated_at,period_type")
                  .eq("status", "complete")
                  .eq("period_type", period_type)
                  .order("generated_at", desc=True)
                  .limit(1).execute())
        if not result.data:
            return True, "ok", None

        existing = result.data[0]
        generated_at = datetime.fromisoformat(existing["generated_at"].replace("Z", "+00:00").replace("+00:00", ""))

        # all_time and last_year: once ever
        if period_type in ("all_time", "last_year"):
            return False, f"Already generated for {PERIOD_LABELS[period_type]}. This report can only be generated once.", existing["id"]

        # last_3_months and last_1_month: once per calendar month
        now = datetime.utcnow()
        if generated_at.year == now.year and generated_at.month == now.month:
            next_month = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
            return False, f"Already generated this month. Next generation available from {next_month.strftime('%1 %B %Y')}.", existing["id"]

        return True, "ok", None
    except Exception as e:
        print(f"[ai] check_generation_allowed error: {e}")
        return True, "ok", None  # Allow on error

def build_energy_summary_for_period(start_date=None, end_date=None, org_id=None):
    """Build energy summary filtered to a date range."""
    elec_query = supabase.table("energy_data").select("timestamp,consumption").range(0, 20000)
    gas_query = supabase.table("gas_data").select("timestamp,consumption").range(0, 20000)
    if org_id: elec_query=elec_query.eq("org_id",org_id); gas_query=gas_query.eq("org_id",org_id)
    if start_date:
        elec_query = elec_query.gte("timestamp", start_date)
        gas_query = gas_query.gte("timestamp", start_date)
    if end_date:
        elec_query = elec_query.lte("timestamp", end_date + "T23:59:59")
        gas_query = gas_query.lte("timestamp", end_date + "T23:59:59")
    elec_data = elec_query.execute().data or []
    gas_data = gas_query.execute().data or []
    summary = {}
    if elec_data:
        df=pd.DataFrame(elec_data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        df["hour"]=df["timestamp"].dt.hour; df["dow"]=df["timestamp"].dt.dayofweek
        df["month"]=df["timestamp"].dt.month; df["date"]=df["timestamp"].dt.date.astype(str)
        df["is_weekend"]=df["dow"]>=5
        daily=df.groupby("date")["consumption"].sum(); monthly=df.groupby("month")["consumption"].sum()
        off_hours=df[df["hour"].isin([22,23,0,1,2,3,4,5,6])]
        hourly_avg=df.groupby("hour")["consumption"].mean()
        wd_avg=df[~df["is_weekend"]].groupby("date")["consumption"].sum().mean()
        we_avg=df[df["is_weekend"]].groupby("date")["consumption"].sum().mean()
        ms=monthly.sort_index(); mom=None
        if len(ms)>=2:
            last,prev=float(ms.iloc[-1]),float(ms.iloc[-2])
            mom=round((last-prev)/prev*100,1) if prev else None
        total_kwh=round(float(df["consumption"].sum()),1)
        days_of_data = int(df["date"].nunique())
        summary["electricity"]={
            "total_kwh":total_kwh,"total_cost_gbp":round(total_kwh*ELECTRICITY_RATE_GBP,2),
            "avg_daily_kwh":round(float(daily.mean()),1),"peak_daily_kwh":round(float(daily.max()),1),
            "min_daily_kwh":round(float(daily.min()),1),
            "baseload_kwh":round(float(df["consumption"].quantile(0.1)),2),
            "peak_demand_kwh":round(float(df["consumption"].max()),2),
            "peak_hour":int(hourly_avg.idxmax()),"quiet_hour":int(hourly_avg.idxmin()),
            "avg_weekday_daily":round(float(wd_avg),1) if not np.isnan(wd_avg) else 0,
            "avg_weekend_daily":round(float(we_avg),1) if not np.isnan(we_avg) else 0,
            "off_hours_avg_kwh":round(float(off_hours["consumption"].mean()),2) if not off_hours.empty else 0,
            "off_hours_pct":round(float(off_hours["consumption"].sum()/df["consumption"].sum()*100),1) if total_kwh else 0,
            "month_on_month_pct":mom,
            "monthly_breakdown":{str(k):round(float(v),1) for k,v in monthly.items()},
            "data_from":str(df["date"].min()),"data_to":str(df["date"].max()),
            "days_of_data":days_of_data
        }
    if gas_data:
        gdf=pd.DataFrame(gas_data); gdf["consumption"]=pd.to_numeric(gdf["consumption"],errors="coerce")
        gdf=gdf.dropna(subset=["consumption"]); gdf=parse_timestamps_naive(gdf)
        gdf["date"]=gdf["timestamp"].dt.date.astype(str); gdf["month"]=gdf["timestamp"].dt.month
        gas_daily=gdf.groupby("date")["consumption"].sum()
        gas_monthly=gdf.groupby("month")["consumption"].sum()
        gas_total=round(float(gdf["consumption"].sum()),1)
        summary["gas"]={
            "total_kwh":gas_total,"total_cost_gbp":round(gas_total*GAS_RATE_GBP,2),
            "avg_daily_kwh":round(float(gas_daily.mean()),1),
            "peak_daily_kwh":round(float(gas_daily.max()),1),
            "monthly_breakdown":{str(k):round(float(v),1) for k,v in gas_monthly.items()},
            "data_from":str(gdf["date"].min()),"data_to":str(gdf["date"].max())
        }
    ec=summary.get("electricity",{}).get("total_cost_gbp",0)
    gc=summary.get("gas",{}).get("total_cost_gbp",0)
    summary["combined"]={
        "total_energy_kwh":round(summary.get("electricity",{}).get("total_kwh",0)+summary.get("gas",{}).get("total_kwh",0),1),
        "total_cost_gbp":round(ec+gc,2)
    }
    return summary

async def run_agentic_analysis(system_prompt: str, user_prompt: str, max_iterations: int = 10, org_id: str = None) -> str:
    """Agentic loop with full tool access — electricity, gas, BMS equipment."""
    messages = [{"role": "user", "content": user_prompt}]
    final_response = ""
    for iteration in range(max_iterations):
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": 6000, "system": system_prompt, "tools": ANALYST_TOOLS, "messages": messages})
            resp.raise_for_status()
            result = resp.json()
        stop_reason = result.get("stop_reason")
        content = result.get("content", [])
        text_parts = [b["text"] for b in content if b["type"] == "text"]
        if text_parts: final_response = "\n".join(text_parts)
        if stop_reason == "end_turn": break
        if stop_reason == "tool_use":
            tool_blocks = [b for b in content if b["type"] == "tool_use"]
            messages.append({"role": "assistant", "content": content})
            tool_results = []
            for tb in tool_blocks:
                print(f"[agentic] Tool: {tb['name']} — {tb['input']}")
                tool_results.append({"type": "tool_result", "tool_use_id": tb["id"], "content": execute_tool(tb["name"], tb["input"], org_id=org_id)})
            messages.append({"role": "user", "content": tool_results})
        else:
            break
    return final_response


async def run_ai_generation(org_id: str = None, period_type: str = "all_time"):
    print(f"[ai] Starting agentic generation — period={period_type} org={org_id}")
    placeholder = supabase.table("ai_insights").insert({
        "status": "generating", "generated_at": datetime.utcnow().isoformat(),
        "period_type": period_type, "org_id": org_id,
    }).execute()
    row_id = placeholder.data[0]["id"] if placeholder.data else None
    try:
        start_date, end_date = get_period_date_range(period_type)
        period_label = PERIOD_LABELS.get(period_type, "the selected period")
        today = datetime.utcnow().strftime("%Y-%m-%d")
        date_range_str = f"{start_date or 'all available'} to {end_date}"

        system_prompt = f"""You are an expert energy analyst generating a structured analysis report.
Today: {today} | Period: {period_label} ({date_range_str})
UK rates: Electricity £{ELECTRICITY_RATE_GBP}/kWh | Gas £{GAS_RATE_GBP}/kWh | Carbon: 0.207 kgCO2/kWh

Use your tools to gather data from ALL sources — electricity, gas, and BMS equipment — then produce a JSON report.

REQUIRED: Call these tools before producing output:
1. get_active_faults — check equipment faults
2. get_site_equipment — list all equipment
3. get_monthly_stats with appropriate year(s)
4. get_gas_data for the period
5. get_anomalies for the period
6. get_equipment_readings for any equipment found

Then return ONLY this JSON:
{{
  "executive_summary": "3-4 sentences mentioning: period ({period_label}), total combined cost, key finding from electricity+gas+BMS data, biggest saving opportunity.",
  "insights": [{{
    "id": "slug", "category": "baseload|peak_demand|off_hours|weekday_weekend|seasonal|gas|cost|trend|equipment|fault",
    "title": "Title", "finding": "Specific finding with numbers from tool calls",
    "implication": "Business impact", "severity": "high|medium|low|positive",
    "audience": ["facilities","consultant","executive"], "metric": "number", "metric_label": "unit"
  }}],
  "recommendations": [{{
    "id": "slug", "title": "Action", "action": "Specific step", "rationale": "Why",
    "saving_kwh_monthly": 0, "saving_gbp_monthly": 0, "effort": "low|medium|high",
    "timeframe": "immediate|1_month|3_months|6_months", "payback_months": 0,
    "category": "behavioural|controls|equipment|monitoring|procurement|bms",
    "audience": ["facilities","consultant","executive"], "priority": "quick_win|medium_term|long_term",
    "data_source": "electricity|gas|bms|combined"
  }}]
}}
Generate 6-10 insights and 6-8 recommendations. Cover all data sources. Flag active faults as high severity. Return ONLY JSON."""

        user_prompt = f"Generate comprehensive energy analysis for: {period_label} ({date_range_str}). Check faults first, then gather all electricity, gas, and BMS equipment data before producing your JSON report."

        raw = await run_agentic_analysis(system_prompt, user_prompt, max_iterations=12, org_id=org_id)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
        start_idx = raw.find("{"); end_idx = raw.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx: raw = raw[start_idx:end_idx]
        result = json.loads(raw)

        payload = {
            "status": "complete", "generated_at": datetime.utcnow().isoformat(),
            "period_type": period_type, "data_from": start_date, "data_to": end_date,
            "executive_summary": result.get("executive_summary", ""),
            "insights": result.get("insights", []), "recommendations": result.get("recommendations", []),
            "raw_stats": {"period_type": period_type, "date_range": date_range_str}
        }
        if row_id: supabase.table("ai_insights").update(payload).eq("id", row_id).execute()
        else: supabase.table("ai_insights").insert(payload).execute()
        print(f"[ai] Done — {len(result.get('insights', []))} insights, {len(result.get('recommendations', []))} recs")
    except Exception as e:
        print(f"[ai] Failed: {e}")
        import traceback; traceback.print_exc()
        if row_id: supabase.table("ai_insights").update({"status": "error", "error_message": str(e)}).eq("id", row_id).execute()


@app.get("/ai/insights/data-availability")
def get_insights_data_availability(org_id: Optional[str]=Query(default=None),
                                    authorization: Optional[str]=Header(default=None)):
    """
    Check what data is available for insights generation.
    Returns availability of electricity, gas, and BMS data,
    plus generation status for each period type.
    """
    require_feature_jwt(authorization, org_id, "ai_insights")
    try:
        # Check electricity data
        elec_q = supabase.table("energy_data").select("timestamp").order("timestamp", desc=False).limit(1)
        elec_latest_q = supabase.table("energy_data").select("timestamp").order("timestamp", desc=True).limit(1)
        if org_id: elec_q=elec_q.eq("org_id",org_id); elec_latest_q=elec_latest_q.eq("org_id",org_id)
        elec = elec_q.execute().data
        elec_latest = elec_latest_q.execute().data
        elec_available = len(elec) > 0
        elec_from = elec[0]["timestamp"][:10] if elec else None
        elec_to = elec_latest[0]["timestamp"][:10] if elec_latest else None

        # Check gas data
        gas_q = supabase.table("gas_data").select("timestamp").order("timestamp", desc=False).limit(1)
        gas_latest_q = supabase.table("gas_data").select("timestamp").order("timestamp", desc=True).limit(1)
        if org_id: gas_q=gas_q.eq("org_id",org_id); gas_latest_q=gas_latest_q.eq("org_id",org_id)
        gas = gas_q.execute().data
        gas_latest = gas_latest_q.execute().data
        gas_available = len(gas) > 0
        gas_from = gas[0]["timestamp"][:10] if gas else None
        gas_to = gas_latest[0]["timestamp"][:10] if gas_latest else None

        # Check BMS data
        bms_count = supabase.table("equipment_readings").select("id", count="exact").limit(1).execute()
        bms_available = (bms_count.count or 0) > 0

        # Check generation status per period
        periods = {}
        for period_type in ["all_time", "last_year", "last_3_months", "last_1_month"]:
            allowed, reason, existing_id = check_generation_allowed(org_id or "", period_type)
            # Get existing insight if any
            existing = None
            if existing_id:
                row = supabase.table("ai_insights").select("id,generated_at,data_from,data_to").eq("id", existing_id).single().execute().data
                if row: existing = {"id": row["id"], "generated_at": row["generated_at"], "data_from": row["data_from"], "data_to": row["data_to"]}
            periods[period_type] = {
                "label": PERIOD_LABELS[period_type],
                "can_generate": allowed,
                "reason": reason if not allowed else None,
                "existing": existing,
            }

        return {
            "electricity": {"available": elec_available, "from": elec_from, "to": elec_to},
            "gas": {"available": gas_available, "from": gas_from, "to": gas_to},
            "bms": {"available": bms_available},
            "periods": periods,
        }
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}

@app.get("/ai/insights")
def get_ai_insights(org_id: Optional[str]=Query(default=None),
                    period_type: Optional[str]=Query(default=None),
                    authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization,org_id,"ai_insights")
    try:
        query = supabase.table("ai_insights").select("*").eq("status","complete").order("generated_at",desc=True)
        if period_type:
            query = query.eq("period_type", period_type)
        result = query.limit(1).execute()
        if not result.data:
            pending=supabase.table("ai_insights").select("id,status,period_type").eq("status","generating").order("generated_at",desc=True).limit(1).execute()
            if pending.data: return {"status":"generating","insights":None,"period_type":pending.data[0].get("period_type")}
            return {"status":"empty","insights":None}
        row=result.data[0]
        return {
            "status":"complete","generatedAt":row["generated_at"],
            "dataFrom":row["data_from"],"dataTo":row["data_to"],
            "periodType":row.get("period_type","all_time"),
            "periodLabel":PERIOD_LABELS.get(row.get("period_type","all_time"),""),
            "executiveSummary":row["executive_summary"],
            "insights":row["insights"] or [],"recommendations":row["recommendations"] or []
        }
    except HTTPException: raise
    except Exception as e: return {"success":False,"message":str(e)}

@app.post("/ai/insights/generate")
async def trigger_ai_generation(background_tasks: BackgroundTasks,
                                 org_id: Optional[str]=Query(default=None),
                                 period_type: str=Query(default="all_time"),
                                 authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization,org_id,"ai_insights")
    if not ANTHROPIC_API_KEY: return {"success":False,"message":"ANTHROPIC_API_KEY not configured"}
    if period_type not in PERIOD_LABELS:
        return {"success":False,"message":f"Invalid period_type. Must be one of: {list(PERIOD_LABELS.keys())}"}

    # Check if generation is already in progress
    in_progress = supabase.table("ai_insights").select("id").eq("status","generating").limit(1).execute()
    if in_progress.data:
        return {"success":False,"message":"Generation already in progress. Please wait."}

    # Check generation rules
    allowed, reason, existing_id = check_generation_allowed(org_id or "", period_type)
    if not allowed:
        return {"success":False,"message":reason,"existing_id":existing_id,"blocked":True}

    background_tasks.add_task(run_ai_generation, org_id=org_id, period_type=period_type)
    return {"success":True,"message":f"Generation started for {PERIOD_LABELS[period_type]}","period_type":period_type}

@app.get("/ai/insights/history")
def get_ai_insights_history(org_id: Optional[str]=Query(default=None)):
    try:
        result=supabase.table("ai_insights").select("id,generated_at,data_from,data_to,status,error_message,period_type").order("generated_at",desc=True).limit(20).execute()
        return {"history":result.data or []}
    except Exception as e: return {"success":False,"message":str(e)}


# ─── AI RECOMMENDATIONS ───────────────────────────────────────

async def run_recommendations_generation(org_id: str = None):
    """
    Generate always-live action recommendations using full data access.
    Covers electricity + gas + BMS equipment.
    Uses last 30 days as the primary window for freshness.
    """
    print(f"[recs] Starting recommendations generation for org={org_id}")
    placeholder = supabase.table("ai_recommendations").insert({
        "status": "generating", "generated_at": datetime.utcnow().isoformat(), "org_id": org_id,
    }).execute()
    row_id = placeholder.data[0]["id"] if placeholder.data else None
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        start_30d = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

        system_prompt = f"""You are an expert energy consultant generating a prioritised action list for a building operator.
Today: {today} | Data window: last 30 days ({start_30d} to {today})
UK rates: Electricity £{ELECTRICITY_RATE_GBP}/kWh | Gas £{GAS_RATE_GBP}/kWh

Use your tools to gather current data from electricity, gas, and BMS equipment, then produce a prioritised action list.

REQUIRED tool calls:
1. get_active_faults — URGENT: any active faults need immediate action
2. get_site_equipment — understand what equipment exists
3. get_daily_summary start={start_30d} end={today} — recent electricity
4. get_gas_data start={start_30d} end={today} — recent gas
5. get_anomalies start={start_30d} end={today} — recent anomalies
6. get_equipment_readings for key equipment — check temperatures, run hours, valve positions

Then return ONLY this JSON:
{{
  "generated_at": "{today}",
  "data_window": "Last 30 days",
  "summary": "2 sentences: current energy spend rate + top priority action right now",
  "quick_wins": [{{
    "id": "slug", "title": "Action title", "action": "Exactly what to do",
    "why": "What data shows this is needed", "saving_gbp_monthly": 0,
    "saving_kwh_monthly": 0, "effort": "low", "can_do_today": true,
    "data_source": "electricity|gas|bms|combined"
  }}],
  "medium_term": [{{
    "id": "slug", "title": "Action title", "action": "Exactly what to do",
    "why": "What data shows this is needed", "saving_gbp_monthly": 0,
    "saving_kwh_monthly": 0, "effort": "medium", "timeframe": "1-3 months",
    "investment_gbp": 0, "payback_months": 0, "data_source": "electricity|gas|bms|combined"
  }}],
  "long_term": [{{
    "id": "slug", "title": "Action title", "action": "Exactly what to do",
    "why": "What data shows this is needed", "saving_gbp_monthly": 0,
    "saving_kwh_monthly": 0, "effort": "high", "timeframe": "3-12 months",
    "investment_gbp": 0, "payback_months": 0, "data_source": "electricity|gas|bms|combined"
  }}],
  "urgent_alerts": [{{
    "id": "slug", "type": "fault|anomaly|threshold",
    "title": "Alert title", "detail": "What was found and why it needs attention now",
    "equipment": "equipment name if applicable", "action": "What to do immediately"
  }}]
}}

Generate 3-5 quick wins, 3-4 medium term, 2-3 long term. Add urgent_alerts for any active faults or anomalies.
Base everything on actual data from your tool calls. Return ONLY JSON."""

        user_prompt = f"Generate a fresh prioritised action list for today ({today}). Check faults first, then review last 30 days of electricity, gas, and BMS equipment data."

        raw = await run_agentic_analysis(system_prompt, user_prompt, max_iterations=10, org_id=org_id)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
        start_idx = raw.find("{"); end_idx = raw.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx: raw = raw[start_idx:end_idx]
        result = json.loads(raw)

        payload = {
            "status": "complete", "generated_at": datetime.utcnow().isoformat(), "org_id": org_id,
            "summary": result.get("summary", ""),
            "quick_wins": result.get("quick_wins", []),
            "medium_term": result.get("medium_term", []),
            "long_term": result.get("long_term", []),
            "urgent_alerts": result.get("urgent_alerts", []),
        }
        if row_id: supabase.table("ai_recommendations").update(payload).eq("id", row_id).execute()
        else: supabase.table("ai_recommendations").insert(payload).execute()
        print(f"[recs] Done — {len(result.get('quick_wins',[]))} quick wins, {len(result.get('urgent_alerts',[]))} alerts")
    except Exception as e:
        print(f"[recs] Failed: {e}")
        import traceback; traceback.print_exc()
        if row_id: supabase.table("ai_recommendations").update({"status": "error", "error_message": str(e)}).eq("id", row_id).execute()


@app.get("/ai/recommendations")
def get_recommendations(org_id: Optional[str]=Query(default=None),
                         authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, org_id, "ai_recommendations")
    try:
        query = supabase.table("ai_recommendations").select("*").eq("status", "complete").order("generated_at", desc=True)
        if org_id: query = query.eq("org_id", org_id)
        result = query.limit(1).execute()
        if not result.data:
            pending = supabase.table("ai_recommendations").select("id,status").eq("status","generating").limit(1).execute()
            if pending.data: return {"status": "generating"}
            return {"status": "empty"}
        row = result.data[0]
        return {
            "status": "complete",
            "generatedAt": row["generated_at"],
            "summary": row.get("summary", ""),
            "quickWins": row.get("quick_wins", []),
            "mediumTerm": row.get("medium_term", []),
            "longTerm": row.get("long_term", []),
            "urgentAlerts": row.get("urgent_alerts", []),
        }
    except HTTPException: raise
    except Exception as e: return {"success": False, "message": str(e)}


@app.post("/ai/recommendations/generate")
async def trigger_recommendations(background_tasks: BackgroundTasks,
                                   org_id: Optional[str]=Query(default=None),
                                   authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, org_id, "ai_recommendations")
    if not ANTHROPIC_API_KEY: return {"success": False, "message": "ANTHROPIC_API_KEY not configured"}
    in_progress = supabase.table("ai_recommendations").select("id").eq("status","generating").limit(1).execute()
    if in_progress.data: return {"success": False, "message": "Generation already in progress"}
    background_tasks.add_task(run_recommendations_generation, org_id=org_id)
    return {"success": True, "message": "Recommendations generation started"}

# ─── AI ANALYST ───────────────────────────────────────────────

@app.post("/ai/analyst/chat")
async def ai_analyst_chat(req: ChatRequest,
                           authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization, req.org_id, "ai_energy_analyst")
    try:
        stats=build_energy_summary_for_ai(org_id=req.org_id)
        elec=stats.get("electricity",{}); gas=stats.get("gas",{})
        today=datetime.utcnow().strftime("%Y-%m-%d")

        # Fetch BMS context for the org's first active site
        site_id = None
        try:
            sites = (supabase.table("sites").select("id").eq("is_active", True).limit(1).execute().data)
            if sites: site_id = sites[0]["id"]
        except Exception: pass
        bms_context = build_bms_context_for_ai(site_id=site_id, days=7)

        tier, _ = resolve_tier(authorization, req.org_id)
        if tier == "enterprise":
            system=build_senior_consultant_prompt(elec, gas, today, bms_context)
            print(f"[analyst] SENIOR CONSULTANT prompt (tier={tier}, bms={'yes' if bms_context else 'no'})")
        else:
            system=build_data_analyst_prompt(elec, gas, today, bms_context)
            print(f"[analyst] DATA ANALYST prompt (tier={tier}, bms={'yes' if bms_context else 'no'})")

        messages=[{"role":m.role,"content":m.content} for m in req.messages]
        original_messages=[{"role":m.role,"content":m.content} for m in req.messages]
        return await run_analyst_chat(messages,system,req.org_id,req.conversation_id,original_messages)

    except HTTPException: raise
    except Exception as e:
        print(f"[analyst] ERROR: {e}"); return {"success":False,"message":str(e)}

@app.get("/ai/analyst/conversations")
def get_conversations(org_id: str=Query(...), authorization: Optional[str]=Header(default=None)):
    require_feature_jwt(authorization,org_id,"ai_energy_analyst")
    try:
        result=supabase.table("ai_conversations").select("id,title,created_at,updated_at").eq("org_id",org_id).order("updated_at",desc=True).limit(20).execute()
        return {"conversations":result.data or []}
    except HTTPException: raise
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/ai/analyst/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    try:
        result=supabase.table("ai_conversations").select("*").eq("id",conversation_id).single().execute()
        return {"conversation":result.data}
    except Exception as e: return {"success":False,"message":str(e)}

# ─── REPORTS ──────────────────────────────────────────────────

def build_report_data(date_from,date_to,report_type,org_id=None):
    elec_q=supabase.table("energy_data").select("timestamp,consumption").gte("timestamp",date_from).lte("timestamp",date_to+"T23:59:59").range(0,20000)
    gas_q=supabase.table("gas_data").select("timestamp,consumption").gte("timestamp",date_from).lte("timestamp",date_to+"T23:59:59").range(0,20000)
    if org_id: elec_q=elec_q.eq("org_id",org_id); gas_q=gas_q.eq("org_id",org_id)
    elec_data=elec_q.execute().data or []
    gas_data=gas_q.execute().data or []
    report={"report_type":report_type,"date_from":date_from,"date_to":date_to,"generated_at":datetime.utcnow().isoformat()}
    if elec_data:
        df=pd.DataFrame(elec_data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        df["date"]=df["timestamp"].dt.date.astype(str); df["month"]=df["timestamp"].dt.month
        daily=df.groupby("date")["consumption"].sum(); monthly=df.groupby("month")["consumption"].sum()
        total_kwh=round(float(df["consumption"].sum()),2)
        report["electricity"]={"total_kwh":total_kwh,"total_cost_gbp":round(total_kwh*ELECTRICITY_RATE_GBP,2),
            "avg_daily_kwh":round(float(daily.mean()),2),"peak_daily_kwh":round(float(daily.max()),2),
            "baseload_kwh":round(float(df["consumption"].quantile(0.1)),2),
            "peak_demand_kwh":round(float(df["consumption"].max()),2),
            "hourly_profile":build_hourly_profile(df),
            "daily_breakdown":[{"date":d,"kwh":round(float(v),2),"cost":round(float(v)*ELECTRICITY_RATE_GBP,2)} for d,v in daily.items()],
            "monthly_breakdown":[{"month":int(m),"kwh":round(float(v),2),"cost":round(float(v)*ELECTRICITY_RATE_GBP,2)} for m,v in monthly.items()]}
    if gas_data:
        gdf=pd.DataFrame(gas_data); gdf["consumption"]=pd.to_numeric(gdf["consumption"],errors="coerce")
        gdf=gdf.dropna(subset=["consumption"]); gdf=parse_timestamps_naive(gdf)
        gdf["date"]=gdf["timestamp"].dt.date.astype(str); gdf["month"]=gdf["timestamp"].dt.month
        gas_daily=gdf.groupby("date")["consumption"].sum(); gas_monthly=gdf.groupby("month")["consumption"].sum()
        gas_total=round(float(gdf["consumption"].sum()),2)
        report["gas"]={"total_kwh":gas_total,"total_cost_gbp":round(gas_total*GAS_RATE_GBP,2),
            "avg_daily_kwh":round(float(gas_daily.mean()),2),
            "daily_breakdown":[{"date":d,"kwh":round(float(v),2),"cost":round(float(v)*GAS_RATE_GBP,2)} for d,v in gas_daily.items()],
            "monthly_breakdown":[{"month":int(m),"kwh":round(float(v),2),"cost":round(float(v)*GAS_RATE_GBP,2)} for m,v in gas_monthly.items()]}
    ek=report.get("electricity",{}).get("total_kwh",0); gk=report.get("gas",{}).get("total_kwh",0)
    ec=report.get("electricity",{}).get("total_cost_gbp",0); gc=report.get("gas",{}).get("total_cost_gbp",0)
    report["combined"]={"total_kwh":round(ek+gk,2),"total_cost_gbp":round(ec+gc,2),
        "electricity_share_pct":round(ek/(ek+gk)*100,1) if (ek+gk) else 0,
        "gas_share_pct":round(gk/(ek+gk)*100,1) if (ek+gk) else 0}
    if report_type in ("ai_insights","full","premium_full"):
        ai_result=supabase.table("ai_insights").select("executive_summary,insights,recommendations").eq("status","complete").order("generated_at",desc=True).limit(1).execute()
        if ai_result.data:
            row=ai_result.data[0]
            report["ai_insights"]={"executive_summary":row["executive_summary"],
                "insights":row["insights"] or [],"recommendations":row["recommendations"] or []}
    return report

@app.post("/reports/generate")
def generate_report(req: ReportRequest, authorization: Optional[str]=Header(default=None)):
    feature_map={"basic":"report_basic","ai_insights":"report_ai_insights","full":"report_full","premium_full":"report_premium_full"}
    require_feature_jwt(authorization,req.org_id,feature_map.get(req.report_type,"report_basic"))
    try:
        data=build_report_data(req.date_from,req.date_to,req.report_type,org_id=req.org_id)
        titles={"basic":"Basic Energy Report","ai_insights":"AI Insights Report","full":"Full Energy Report","premium_full":"Premium Full Report"}
        title=f"{titles.get(req.report_type,'Report')} — {req.date_from} to {req.date_to}"
        result=supabase.table("reports").insert({"org_id":req.org_id,"report_type":req.report_type,"title":title,
            "date_from":req.date_from,"date_to":req.date_to,"period_type":req.period_type,"status":"complete","data":data}).execute()
        return {"success":True,"report_id":result.data[0]["id"] if result.data else None,"title":title,"data":data}
    except HTTPException: raise
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/reports")
def list_reports(org_id: Optional[str]=Query(default=None)):
    try:
        query=supabase.table("reports").select("id,report_type,title,date_from,date_to,period_type,status,created_at")
        if org_id: query=query.eq("org_id",org_id)
        result=query.order("created_at",desc=True).limit(50).execute()
        return {"reports":result.data or []}
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/reports/{report_id}")
def get_report(report_id: str):
    try:
        result=supabase.table("reports").select("*").eq("id",report_id).single().execute()
        if not result.data: return {"success":False,"message":"Report not found"}
        return {"report":result.data}
    except Exception as e: return {"success":False,"message":str(e)}

@app.delete("/reports/{report_id}")
def delete_report(report_id: str):
    try: supabase.table("reports").delete().eq("id",report_id).execute(); return {"success":True,"deleted":report_id}
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/reports/preview/{report_type}")
def preview_report(report_type: str,date_from: str=Query(...),date_to: str=Query(...),
                   org_id: Optional[str]=Query(default=None),authorization: Optional[str]=Header(default=None)):
    feature_map={"basic":"report_basic","ai_insights":"report_ai_insights","full":"report_full","premium_full":"report_premium_full"}
    require_feature_jwt(authorization,org_id,feature_map.get(report_type,"report_basic"))
    try: return {"success":True,"data":build_report_data(date_from,date_to,report_type,org_id=org_id)}
    except HTTPException: raise
    except Exception as e: return {"success":False,"message":str(e)}

# ─── UPLOAD / ANALYTICS ───────────────────────────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile=File(...), org_id: Optional[str]=Query(default=None), authorization: Optional[str]=Header(default=None)):
    try:
        contents=await file.read(); df=pd.read_csv(StringIO(contents.decode("utf-8")),index_col=None)
        date_col=next((c for c in df.columns if "date" in c.lower()),None)
        if not date_col: return {"success":False,"message":"No date column"}
        time_columns=[col for col in df.columns if ":" in col]
        if not time_columns: return {"success":False,"message":"No time columns"}
        df_long=df.melt(id_vars=[date_col],value_vars=time_columns,var_name="time",value_name="consumption")
        df_long["consumption"]=pd.to_numeric(df_long["consumption"],errors="coerce")
        df_long=df_long.dropna(subset=["consumption"])
        df_long["timestamp"]=pd.to_datetime(df_long[date_col].astype(str)+" "+df_long["time"].astype(str),dayfirst=True,errors="coerce")
        df_long=df_long.dropna(subset=["timestamp"])
        df_agg=df_long.groupby("timestamp",as_index=False)["consumption"].sum()
        if df_agg.empty: return {"success":False,"message":"No valid data"}
        df_agg["timestamp"]=df_agg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_agg["consumption"]=df_agg["consumption"].astype(float)
        records=df_agg.to_dict(orient="records")
        resolved_org_id = org_id
        if not resolved_org_id and authorization:
            try:
                _, org = require_auth(authorization)
                if org: resolved_org_id = org.get("id")
            except: pass
        if resolved_org_id:
            for r in records: r["org_id"] = resolved_org_id
        for i in range(0,len(records),500): supabase.table("energy_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Data stored"}
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/analytics")
def get_analytics(org_id: Optional[str]=Query(default=None), authorization: Optional[str]=Header(default=None)):
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    q=supabase.table("energy_data").select("*").range(0,20000)
    if resolved_org_id: q=q.eq("org_id",resolved_org_id)
    data=q.execute().data
    if not data:
        return {"stats":{"baseload":0,"peakDemand":0,"loadFactor":0,"avgDaily":0},
                "hourlyProfile":[],"daily":[],"totalConsumption":0,"heatmap":[[0.0]*24 for _ in range(7)]}
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df["hour"]=df["timestamp"].dt.hour; df["is_weekend"]=df["timestamp"].dt.dayofweek>=5
    df["date"]=df["timestamp"].dt.date
    stats=build_stats(df); total=round(float(df["consumption"].sum()),2)
    daily=df.groupby("date")["consumption"].sum().reset_index(); daily_breakdown=[]
    for _,row in daily.iterrows():
        dv=row["date"]; dd=df[df["date"]==dv]
        daily_breakdown.append({"date":str(dv),"consumption":round(float(row["consumption"]),2),
            "hourly":[round(float(dd[dd["hour"]==h]["consumption"].sum()),2) for h in range(24)]})
    heatmap=[[0.0]*24 for _ in range(7)]; counts=[[0]*24 for _ in range(7)]
    for _,row in df.iterrows():
        d,h=row["timestamp"].dayofweek,row["timestamp"].hour
        heatmap[d][h]+=row["consumption"]; counts[d][h]+=1
    for d in range(7):
        for h in range(24):
            if counts[d][h]>0: heatmap[d][h]=round(heatmap[d][h]/counts[d][h],2)
    return {"stats":stats,"hourlyProfile":build_hourly_profile(df),"daily":daily_breakdown,
            "totalConsumption":total,"heatmap":heatmap}

@app.get("/analytics/hourly-profile/{year}")
def get_hourly_profile_by_year(year: int, org_id: Optional[str]=Query(default=None), authorization: Optional[str]=Header(default=None)):
    try:
        resolved_org_id = org_id
        if not resolved_org_id and authorization:
            try:
                _, org = require_auth(authorization)
                if org: resolved_org_id = org.get("id")
            except: pass
        q=supabase.table("energy_data").select("timestamp,consumption").gte("timestamp",f"{year}-01-01").lte("timestamp",f"{year}-12-31T23:59:59")
        if resolved_org_id: q=q.eq("org_id",resolved_org_id)
        data=q.execute().data
        if not data: return {"hourlyProfile":[{"hour":f"{h:02d}:00","average":0,"weekday":0,"weekend":0} for h in range(24)]}
        df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        return {"hourlyProfile":build_hourly_profile(df)}
    except Exception as e: return {"success":False,"message":str(e)}

@app.post("/upload-gas-data")
async def upload_gas_data(file: UploadFile=File(...), org_id: Optional[str]=Query(default=None), authorization: Optional[str]=Header(default=None)):
    try:
        contents=await file.read(); df=pd.read_csv(StringIO(contents.decode("utf-8")),index_col=None)
        date_col=next((c for c in df.columns if "date" in c.lower()),None)
        if not date_col: return {"success":False,"message":"No date column"}
        time_columns=[col for col in df.columns if ":" in col]
        if not time_columns: return {"success":False,"message":"No time columns"}
        df_long=df.melt(id_vars=[date_col],value_vars=time_columns,var_name="time",value_name="consumption")
        df_long["consumption"]=pd.to_numeric(df_long["consumption"],errors="coerce")
        df_long=df_long.dropna(subset=["consumption"])
        df_long["timestamp"]=pd.to_datetime(df_long[date_col].astype(str)+" "+df_long["time"].astype(str),dayfirst=True,errors="coerce")
        df_long=df_long.dropna(subset=["timestamp"])
        df_agg=df_long.groupby("timestamp",as_index=False)["consumption"].sum()
        if df_agg.empty: return {"success":False,"message":"No valid data"}
        df_agg["timestamp"]=df_agg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        df_agg["consumption"]=df_agg["consumption"].astype(float)
        records=df_agg.to_dict(orient="records")
        resolved_org_id = org_id
        if not resolved_org_id and authorization:
            try:
                _, org = require_auth(authorization)
                if org: resolved_org_id = org.get("id")
            except: pass
        if resolved_org_id:
            for r in records: r["org_id"] = resolved_org_id
        for i in range(0,len(records),500): supabase.table("gas_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Gas data stored"}
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/gas-analytics")
def get_gas_analytics(org_id: Optional[str]=Query(default=None), authorization: Optional[str]=Header(default=None)):
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    q=supabase.table("gas_data").select("*").range(0,20000)
    if resolved_org_id: q=q.eq("org_id",resolved_org_id)
    data=q.execute().data
    if not data:
        return {"stats":{"baseload":0,"peakDemand":0,"loadFactor":0,"avgDaily":0},
                "hourlyProfile":[],"daily":[],"totalConsumption":0}
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df); df["date"]=df["timestamp"].dt.date
    daily=df.groupby("date")["consumption"].sum().reset_index()
    return {"stats":build_stats(df),"hourlyProfile":build_hourly_profile(df),
            "daily":[{"date":str(r["date"]),"consumption":round(float(r["consumption"]),2)} for _,r in daily.iterrows()],
            "totalConsumption":round(float(df["consumption"].sum()),2)}

# ─── DEBUG / DELETE ───────────────────────────────────────────

@app.get("/debug/data-summary")
def debug_data_summary(org_id: Optional[str]=Query(default=None)):
    try:
        q=supabase.table("energy_data").select("*").range(0,20000)
        if org_id: q=q.eq("org_id",org_id)
        data=q.execute().data
        if not data: return {"rowCount":0}
        df=pd.DataFrame(data); df=parse_timestamps_naive(df)
        df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce"); df=df.dropna(subset=["consumption"])
        df["date"]=df["timestamp"].dt.date
        hour_dist=df.groupby(df["timestamp"].dt.hour)["consumption"].count().to_dict()
        unique_dates=df["date"].nunique(); total=float(df["consumption"].sum())
        return {"rowCount":len(df),"dateRange":{"earliest":str(df["timestamp"].min()),"latest":str(df["timestamp"].max())},
                "totalConsumption":round(total,2),"avgConsumption":round(float(df["consumption"].mean()),2),
                "minConsumption":round(float(df["consumption"].min()),2),"maxConsumption":round(float(df["consumption"].max()),2),
                "peakDemand":round(float(df["consumption"].max()),2),
                "avgDaily":round(total/unique_dates,2) if unique_dates else 0,"hourDistribution":hour_dist}
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/debug/gas-summary")
def debug_gas_summary(org_id: Optional[str]=Query(default=None)):
    try:
        q=supabase.table("gas_data").select("*").range(0,20000)
        if org_id: q=q.eq("org_id",org_id)
        data=q.execute().data
        if not data: return {"rowCount":0}
        df=pd.DataFrame(data); df=parse_timestamps_naive(df)
        df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce"); df=df.dropna(subset=["consumption"])
        df["date"]=df["timestamp"].dt.date
        hour_dist=df.groupby(df["timestamp"].dt.hour)["consumption"].count().to_dict()
        unique_dates=df["date"].nunique(); total=float(df["consumption"].sum())
        return {"rowCount":len(df),"dateRange":{"earliest":str(df["timestamp"].min()),"latest":str(df["timestamp"].max())},
                "totalConsumption":round(total,2),"avgConsumption":round(float(df["consumption"].mean()),2),
                "minConsumption":round(float(df["consumption"].min()),2),"maxConsumption":round(float(df["consumption"].max()),2),
                "peakDemand":round(float(df["consumption"].max()),2),
                "avgDaily":round(total/unique_dates,2) if unique_dates else 0,"hourDistribution":hour_dist}
    except Exception as e: return {"success":False,"message":str(e)}

@app.delete("/delete-data")
def delete_data():
    try: supabase.table("energy_data").delete().gt("id","00000000-0000-0000-0000-000000000000").execute(); return {"success":True,"message":"All energy data deleted"}
    except Exception as e: return {"success":False,"message":str(e)}

@app.delete("/delete-gas-data")
def delete_gas_data():
    try: supabase.table("gas_data").delete().gt("id","00000000-0000-0000-0000-000000000000").execute(); return {"success":True,"message":"All gas data deleted"}
    except Exception as e: return {"success":False,"message":str(e)}
