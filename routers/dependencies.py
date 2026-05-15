"""
dependencies.py — Shared dependencies, clients, constants and helpers for Effictra AI
Import from here in main.py, reports.py, carbon.py
"""
import os, pandas as pd
from supabase import create_client
from datetime import datetime, timedelta
from fastapi import HTTPException, Header
from typing import Optional
from pydantic import BaseModel

# ── Environment ───────────────────────────────────────────────────────────────
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY         = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL      = "claude-opus-4-5"
FRONTEND_URL         = os.environ.get("FRONTEND_URL", "https://ai.effictraenergy.co.uk")
ELECTRICITY_RATE_GBP = 0.28
GAS_RATE_GBP         = 0.07

# ── Supabase clients ──────────────────────────────────────────────────────────
supabase         = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ── Pydantic models ───────────────────────────────────────────────────────────
class ReportRequest(BaseModel):
    report_type: str; date_from: str; date_to: str
    period_type: str = "custom"; org_id: Optional[str] = None; site_id: Optional[str] = None

# ── Tier features (shared across auth and feature gates) ──────────────────────
TIER_FEATURES = {
    "trial": {
        "dashboard": True, "analytics": True, "anomalies": True, "upload_data": True,
        "ai_insights": True, "ai_recommendations": True, "ai_energy_analyst": True,
        "ai_senior_consultant": False, "weather_normalisation": True,
        "report_basic": True, "report_ai_insights": True, "report_full": True,
        "report_premium_full": True, "settings_sites": True, "multi_site": False,
        "api_access": False, "bms_parameters": True, "carbon_reporting": True,
    }

FEATURE_REQUIRED_TIER = {
    "ai_insights":           "standard",
    "ai_recommendations":    "standard",
    "weather_normalisation": "standard",
    "report_ai_insights":    "standard",
    "report_full":           "standard",
    "multi_site":            "standard",
    "carbon_reporting":      "standard",
    "ai_energy_analyst":     "premium",
    "report_premium_full":   "premium",
    "api_access":            "premium",
    "bms_parameters":        "premium",
    "ai_senior_consultant":  "enterprise",
}


# ── Auth helpers ──────────────────────────────────────────────────────────────
def get_user_from_token(authorization):
    if not authorization or not authorization.startswith("Bearer "): return None
    token = authorization.replace("Bearer ", "")
    try:
        result = supabase_service.auth.get_user(token)
        return result.user if result else None
    except Exception: return None

def get_org_for_user(auth_id):
    try:
        result = (supabase_service.table("users").select("org_id, role, organisations(*)")
                  .eq("auth_id", auth_id).single().execute())
        if not result.data: return None
        org_data = result.data.get("organisations")
        if isinstance(org_data, list): org_data = org_data[0] if org_data else None
        return org_data
    except Exception: return None

def get_effective_tier(org):
    if not org: return "basic"
    tier = org.get("tier", "basic")
    if tier == "custom": return "enterprise"
    return tier

def get_org_tier_by_id(org_id):
    if not org_id: return "basic"
    try:
        result = supabase_service.table("organisations").select("tier").eq("id", org_id).single().execute()
        if not result.data: return "basic"
        tier = result.data.get("tier", "basic")
        if tier == "custom": return "enterprise"
        if tier not in TIER_FEATURES: tier = "basic"
        return tier
    except Exception: return "basic"

def resolve_tier(authorization, org_id):
    user = get_user_from_token(authorization)
    if user:
        org = get_org_for_user(user.id)
        if org: return get_effective_tier(org), org
    if org_id:
        tier = get_org_tier_by_id(org_id)
        return tier, None
    return "basic", None

def require_auth(authorization):
    user = get_user_from_token(authorization)
    if not user: raise HTTPException(status_code=401, detail="Unauthorized")
    org = get_org_for_user(user.id)
    return user, org

def require_feature_jwt(authorization, org_id, feature):
    tier, org = resolve_tier(authorization, org_id)
    print(f"[feature_gate] feature={feature} org_id={org_id} resolved_tier={tier} org={org}")
    if tier == "custom": tier = "enterprise"
    features = TIER_FEATURES.get(tier, {})
    if not features.get(feature, False):
        required = FEATURE_REQUIRED_TIER.get(feature, "standard")
        raise HTTPException(status_code=403, detail={
            "error":"upgrade_required", "message":f"This feature requires the {required} plan.",
            "current_tier":tier, "required_tier":required, "upgrade_url":f"{FRONTEND_URL}/pricing"
        })

# ── Data helpers ──────────────────────────────────────────────────────────────
def parse_timestamps_naive(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=["timestamp"])

def build_hourly_profile(df):
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekday"] = df["timestamp"].dt.dayofweek < 5
    profile = []
    for h in range(24):
        hdf = df[df["hour"]==h]
        profile.append({"hour":f"{h:02d}:00",
            "average":round(float(hdf["consumption"].mean()),2) if len(hdf) else 0,
            "weekday":round(float(hdf[hdf["is_weekday"]]["consumption"].mean()),2) if len(hdf[hdf["is_weekday"]]) else 0,
            "weekend":round(float(hdf[~hdf["is_weekday"]]["consumption"].mean()),2) if len(hdf[~hdf["is_weekday"]]) else 0})
    return profile

def get_period_date_range(period_type: str, anchor_date: str = None):
    if anchor_date:
        try: end_dt = datetime.strptime(anchor_date, "%Y-%m-%d").date()
        except: end_dt = datetime.utcnow().date()
    else:
        end_dt = datetime.utcnow().date()
    end = str(end_dt)
    if period_type == "all_time": return None, end
    elif period_type == "last_year": return str(end_dt - timedelta(days=365)), end
    elif period_type == "last_3_months": return str(end_dt - timedelta(days=90)), end
    elif period_type == "last_1_month": return str(end_dt - timedelta(days=30)), end
    return None, end
