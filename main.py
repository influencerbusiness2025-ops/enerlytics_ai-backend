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

SUPABASE_URL = "https://fopzbnloivgxzupxvhcr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvcHpibmxvaXZneHp1cHh2aGNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQ5Nzk5ODcsImV4cCI6MjA5MDU1NTk4N30.GC0Rs6N79vcXuyVBCqpyS5xH76sJ-Ea2CrY22gPyDMs"
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
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://your-lovable-app.lovable.app")

ELECTRICITY_RATE_GBP = 0.28
GAS_RATE_GBP         = 0.07

supabase         = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_service = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ─── APP ──────────────────────────────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

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
    plan: str            # basic | standard | premium
    billing_period: str  # monthly | annual
    org_id: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

# ─── TIER CONFIG ──────────────────────────────────────────────

TIER_FEATURES = {
    "trial":      ["dashboard","analytics","anomalies","upload_data","ai_insights",
                   "ai_recommendations","ai_energy_analyst","weather_normalisation",
                   "report_basic","report_ai_insights","report_full","report_premium_full",
                   "settings_sites","multi_site","api_access"],
    "basic":      ["dashboard","analytics","anomalies","upload_data",
                   "report_basic","settings_sites"],
    "standard":   ["dashboard","analytics","anomalies","upload_data",
                   "ai_insights","ai_recommendations","weather_normalisation",
                   "report_basic","report_ai_insights","report_full",
                   "settings_sites","multi_site"],
    "premium":    ["dashboard","analytics","anomalies","upload_data",
                   "ai_insights","ai_recommendations","ai_energy_analyst","weather_normalisation",
                   "report_basic","report_ai_insights","report_full","report_premium_full",
                   "settings_sites","multi_site","api_access"],
    "enterprise": ["dashboard","analytics","anomalies","upload_data",
                   "ai_insights","ai_recommendations","ai_energy_analyst","weather_normalisation",
                   "report_basic","report_ai_insights","report_full","report_premium_full",
                   "settings_sites","multi_site","api_access"],
}

FEATURE_REQUIRED_TIER = {
    "ai_insights":         "standard",
    "ai_recommendations":  "standard",
    "weather_normalisation":"standard",
    "report_ai_insights":  "standard",
    "report_full":         "standard",
    "ai_energy_analyst":   "premium",
    "report_premium_full": "premium",
    "multi_site":          "standard",
    "api_access":          "premium",
}

STRIPE_PRICE_MAP = {
    ("basic",    "monthly"): STRIPE_PRICE_BASIC_MONTHLY,
    ("basic",    "annual"):  STRIPE_PRICE_BASIC_ANNUAL,
    ("standard", "monthly"): STRIPE_PRICE_STANDARD_MONTHLY,
    ("standard", "annual"):  STRIPE_PRICE_STANDARD_ANNUAL,
    ("premium",  "monthly"): STRIPE_PRICE_PREMIUM_MONTHLY,
    ("premium",  "annual"):  STRIPE_PRICE_PREMIUM_ANNUAL,
}

# ─── AUTH HELPERS ─────────────────────────────────────────────

def get_user_from_token(authorization: Optional[str]) -> Optional[dict]:
    """Verify Supabase JWT and return user record."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.replace("Bearer ", "")
    try:
        result = supabase_service.auth.get_user(token)
        return result.user if result else None
    except Exception:
        return None

def get_org_for_user(auth_id: str) -> Optional[dict]:
    """Get organisation for a user by auth_id."""
    try:
        result = (supabase.table("users").select("org_id, role, organisations(*)")
                  .eq("auth_id", auth_id).single().execute())
        if not result.data:
            return None
        org = result.data.get("organisations", {})
        # Check trial expiry
        if org.get("tier") == "trial":
            trial_expires = org.get("trial_expires_at")
            if trial_expires:
                from datetime import timezone
                expires_dt = datetime.fromisoformat(trial_expires.replace("Z", "+00:00"))
                if expires_dt < datetime.now(timezone.utc):
                    # Expire the trial
                    supabase.table("organisations").update({"tier": "basic"}).eq("id", org["id"]).execute()
                    org["tier"] = "basic"
        return org
    except Exception as e:
        print(f"[auth] get_org_for_user error: {e}")
        return None

def get_effective_tier(org: Optional[dict]) -> str:
    if not org:
        return "basic"
    return org.get("tier", "basic")

def require_auth(authorization: Optional[str]) -> tuple:
    """Returns (auth_user, org) or raises 401."""
    auth_user = get_user_from_token(authorization)
    if not auth_user:
        raise HTTPException(status_code=401, detail={"error": "unauthorized", "message": "Please log in."})
    org = get_org_for_user(auth_user.id)
    return auth_user, org

def require_feature_auth(authorization: Optional[str], feature: str) -> tuple:
    """Returns (auth_user, org) or raises 401/403."""
    auth_user, org = require_auth(authorization)
    tier = get_effective_tier(org)
    if feature not in TIER_FEATURES.get(tier, []):
        required = FEATURE_REQUIRED_TIER.get(feature, "standard")
        raise HTTPException(status_code=403, detail={
            "error": "upgrade_required",
            "message": f"This feature requires the {required} plan.",
            "current_tier": tier,
            "required_tier": required,
            "upgrade_url": f"{FRONTEND_URL}/pricing",
        })
    return auth_user, org

# ─── LEGACY TIER CHECK (for non-auth endpoints during dev) ────

def get_org_tier_by_id(org_id: Optional[str]) -> str:
    if not org_id:
        return "premium"  # dev default
    try:
        result = supabase.table("organisations").select("tier, trial_expires_at").eq("id", org_id).single().execute()
        if not result.data:
            return "basic"
        org = result.data
        if org.get("tier") == "trial":
            from datetime import timezone
            trial_expires = org.get("trial_expires_at")
            if trial_expires:
                expires_dt = datetime.fromisoformat(trial_expires.replace("Z", "+00:00"))
                if expires_dt < datetime.now(timezone.utc):
                    supabase.table("organisations").update({"tier": "basic"}).eq("id", org_id).execute()
                    return "basic"
        return org.get("tier", "basic")
    except Exception:
        return "basic"

def require_feature(org_id: Optional[str], feature: str):
    tier = get_org_tier_by_id(org_id)
    if feature not in TIER_FEATURES.get(tier, []):
        required = FEATURE_REQUIRED_TIER.get(feature, "standard")
        raise HTTPException(status_code=403, detail={
            "error": "upgrade_required",
            "message": f"This feature requires the {required} plan.",
            "current_tier": tier, "required_tier": required,
            "upgrade_url": f"{FRONTEND_URL}/pricing",
        })

# ─── ENERGY HELPERS ───────────────────────────────────────────

def parse_timestamps_naive(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"])

def build_hourly_profile(df):
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    profile = []
    for h in range(24):
        hdf = df[df["hour"] == h]
        if hdf.empty:
            profile.append({"hour": f"{h:02d}:00", "average": 0, "weekday": 0, "weekend": 0})
            continue
        avg = hdf["consumption"].mean()
        wd  = hdf[~hdf["is_weekend"]]["consumption"].mean() if not hdf[~hdf["is_weekend"]].empty else avg
        we  = hdf[hdf["is_weekend"]]["consumption"].mean()  if not hdf[hdf["is_weekend"]].empty  else avg
        profile.append({"hour": f"{h:02d}:00", "average": round(float(avg),2),
                        "weekday": round(float(wd),2), "weekend": round(float(we),2)})
    return profile

def build_stats(df):
    peak = df["consumption"].max()
    avg  = df["consumption"].mean()
    return {"baseload": round(float(df["consumption"].quantile(0.1)),2),
            "peakDemand": round(float(peak),2),
            "loadFactor": round(float(avg/peak),2) if peak else 0,
            "avgDaily": round(float(avg*24),2)}

def auto_base_temp(lat, base):
    if base == 15.5 and abs(lat) < 23.5:
        return 24.0 if abs(lat) < 15 else 18.0
    return base

def resolve_mode(mode, lat, base_temp):
    if mode != "auto": return mode
    if base_temp >= 22 or abs(lat) < 15: return "cdd_only"
    if base_temp >= 18 or abs(lat) < 23.5: return "cdd_only"
    return "both"

async def fetch_degree_days(lat, lng, base_temp, start_date, end_date):
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lng}"
           f"&start_date={start_date}&end_date={end_date}&daily=temperature_2m_mean&timezone=UTC")
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url); r.raise_for_status(); data = r.json()
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

async def call_claude(prompt, max_tokens=4000):
    if not ANTHROPIC_API_KEY: raise ValueError("ANTHROPIC_API_KEY not set")
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_MODEL, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"]

async def call_claude_messages(messages, system="", max_tokens=2000):
    if not ANTHROPIC_API_KEY: raise ValueError("ANTHROPIC_API_KEY not set")
    body = {"model": ANTHROPIC_MODEL, "max_tokens": max_tokens, "messages": messages}
    if system: body["system"] = system
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"}, json=body)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

def build_energy_summary_for_ai():
    elec_data = (supabase.table("energy_data").select("timestamp, consumption").range(0,20000).execute().data) or []
    gas_data  = (supabase.table("gas_data").select("timestamp, consumption").range(0,20000).execute().data) or []
    summary = {}
    if elec_data:
        df = pd.DataFrame(elec_data)
        df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
        df = df.dropna(subset=["consumption"])
        df = parse_timestamps_naive(df)
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
        summary["electricity"]={"total_kwh":total_kwh,"total_cost_gbp":round(total_kwh*ELECTRICITY_RATE_GBP,2),
            "avg_daily_kwh":round(float(daily.mean()),1),"peak_daily_kwh":round(float(daily.max()),1),
            "min_daily_kwh":round(float(daily.min()),1),"baseload_kwh":round(float(df["consumption"].quantile(0.1)),2),
            "peak_demand_kwh":round(float(df["consumption"].max()),2),
            "peak_hour":int(hourly_avg.idxmax()),"quiet_hour":int(hourly_avg.idxmin()),
            "avg_weekday_daily":round(float(wd_avg),1) if not np.isnan(wd_avg) else 0,
            "avg_weekend_daily":round(float(we_avg),1) if not np.isnan(we_avg) else 0,
            "off_hours_avg_kwh":round(float(off_hours["consumption"].mean()),2) if not off_hours.empty else 0,
            "off_hours_pct":round(float(off_hours["consumption"].sum()/df["consumption"].sum()*100),1) if total_kwh else 0,
            "month_on_month_pct":mom,"monthly_breakdown":{str(k):round(float(v),1) for k,v in monthly.items()},
            "data_from":str(df["date"].min()),"data_to":str(df["date"].max()),"days_of_data":int(df["date"].nunique())}
    if gas_data:
        gdf=pd.DataFrame(gas_data); gdf["consumption"]=pd.to_numeric(gdf["consumption"],errors="coerce")
        gdf=gdf.dropna(subset=["consumption"]); gdf=parse_timestamps_naive(gdf)
        gdf["date"]=gdf["timestamp"].dt.date.astype(str); gdf["month"]=gdf["timestamp"].dt.month
        gas_daily=gdf.groupby("date")["consumption"].sum(); gas_monthly=gdf.groupby("month")["consumption"].sum()
        gas_total=round(float(gdf["consumption"].sum()),1)
        summary["gas"]={"total_kwh":gas_total,"total_cost_gbp":round(gas_total*GAS_RATE_GBP,2),
            "avg_daily_kwh":round(float(gas_daily.mean()),1),"peak_daily_kwh":round(float(gas_daily.max()),1),
            "monthly_breakdown":{str(k):round(float(v),1) for k,v in gas_monthly.items()},
            "data_from":str(gdf["date"].min()),"data_to":str(gdf["date"].max())}
    ec=summary.get("electricity",{}).get("total_cost_gbp",0)
    gc=summary.get("gas",{}).get("total_cost_gbp",0)
    summary["combined"]={"total_energy_kwh":round(summary.get("electricity",{}).get("total_kwh",0)+summary.get("gas",{}).get("total_kwh",0),1),
        "total_cost_gbp":round(ec+gc,2),"electricity_rate":f"£{ELECTRICITY_RATE_GBP}/kWh","gas_rate":f"£{GAS_RATE_GBP}/kWh"}
    return summary

# ─── ROOT ─────────────────────────────────────────────────────

@app.get("/")
def root(): return {"message": "Enerlytics API running"}

@app.get("/health")
def health(): return {"status": "ok"}

# ─── AUTH ENDPOINTS ───────────────────────────────────────────

@app.get("/auth/me")
def get_me(authorization: Optional[str] = Header(default=None)):
    """Return current user + org + tier."""
    auth_user, org = require_auth(authorization)
    tier = get_effective_tier(org)
    return {
        "user": {"id": str(auth_user.id), "email": auth_user.email},
        "org": org,
        "tier": tier,
        "features": TIER_FEATURES.get(tier, []),
        "trial_expires_at": org.get("trial_expires_at") if org else None,
    }

@app.get("/tier")
def get_tier(
    org_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    """Get tier and features. Supports both JWT auth and org_id param."""
    if authorization and authorization.startswith("Bearer "):
        try:
            auth_user, org = require_auth(authorization)
            tier = get_effective_tier(org)
            return {"tier": tier, "features": TIER_FEATURES.get(tier, []),
                    "trial_expires_at": org.get("trial_expires_at") if org else None,
                    "upgrade_url": f"{FRONTEND_URL}/pricing"}
        except HTTPException:
            pass
    # Fallback to org_id param
    tier = get_org_tier_by_id(org_id)
    return {"tier": tier, "features": TIER_FEATURES.get(tier, []),
            "upgrade_url": f"{FRONTEND_URL}/pricing"}

@app.get("/feature-flags")
def get_feature_flags(
    org_id: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None)
):
    tier = "basic"
    if authorization and authorization.startswith("Bearer "):
        try:
            _, org = require_auth(authorization)
            tier = get_effective_tier(org)
        except HTTPException:
            tier = get_org_tier_by_id(org_id)
    else:
        tier = get_org_tier_by_id(org_id)
    allowed = TIER_FEATURES.get(tier, [])
    all_features = set(f for fl in TIER_FEATURES.values() for f in fl)
    return {"tier": tier, "flags": {f: f in allowed for f in all_features}}

# ─── STRIPE ENDPOINTS ─────────────────────────────────────────

@app.post("/billing/create-checkout")
async def create_checkout_session(req: CheckoutRequest,
                                   authorization: Optional[str] = Header(default=None)):
    """Create a Stripe checkout session for plan upgrade."""
    if not STRIPE_SECRET_KEY:
        return {"success": False, "message": "Stripe not configured"}

    price_id = STRIPE_PRICE_MAP.get((req.plan, req.billing_period))
    if not price_id:
        return {"success": False, "message": f"No price configured for {req.plan}/{req.billing_period}"}

    # Get or create Stripe customer
    org_result = supabase.table("organisations").select("*").eq("id", req.org_id).single().execute()
    if not org_result.data:
        return {"success": False, "message": "Organisation not found"}
    org = org_result.data

    success_url = req.success_url or f"{FRONTEND_URL}/dashboard?upgrade=success"
    cancel_url  = req.cancel_url  or f"{FRONTEND_URL}/pricing?upgrade=cancelled"

    try:
        async with httpx.AsyncClient() as client:
            # Create customer if doesn't exist
            customer_id = org.get("stripe_customer_id")
            if not customer_id:
                user_result = (supabase.table("users").select("email, name")
                               .eq("org_id", req.org_id).limit(1).execute())
                email = user_result.data[0]["email"] if user_result.data else ""
                name  = user_result.data[0]["name"]  if user_result.data else ""
                cust_resp = await client.post(
                    "https://api.stripe.com/v1/customers",
                    auth=(STRIPE_SECRET_KEY, ""),
                    data={"email": email, "name": name,
                          "metadata[org_id]": req.org_id}
                )
                cust_resp.raise_for_status()
                customer_id = cust_resp.json()["id"]
                supabase.table("organisations").update(
                    {"stripe_customer_id": customer_id}).eq("id", req.org_id).execute()

            # Create checkout session
            checkout_data = {
                "customer": customer_id,
                "mode": "subscription",
                "line_items[0][price]": price_id,
                "line_items[0][quantity]": "1",
                "success_url": success_url + "&session_id={CHECKOUT_SESSION_ID}",
                "cancel_url": cancel_url,
                "metadata[org_id]": req.org_id,
                "metadata[plan]": req.plan,
                "subscription_data[metadata][org_id]": req.org_id,
                "subscription_data[metadata][plan]": req.plan,
                "allow_promotion_codes": "true",
            }
            resp = await client.post(
                "https://api.stripe.com/v1/checkout/sessions",
                auth=(STRIPE_SECRET_KEY, ""),
                data=checkout_data
            )
            resp.raise_for_status()
            session = resp.json()
            return {"success": True, "checkout_url": session["url"], "session_id": session["id"]}

    except Exception as e:
        print(f"[stripe] Checkout error: {e}")
        return {"success": False, "message": str(e)}


@app.post("/billing/create-portal")
async def create_billing_portal(org_id: str = Query(...)):
    """Create a Stripe customer portal session for managing subscription."""
    if not STRIPE_SECRET_KEY:
        return {"success": False, "message": "Stripe not configured"}
    try:
        org = supabase.table("organisations").select("stripe_customer_id").eq("id", org_id).single().execute().data
        if not org or not org.get("stripe_customer_id"):
            return {"success": False, "message": "No Stripe customer found"}
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.stripe.com/v1/billing_portal/sessions",
                auth=(STRIPE_SECRET_KEY, ""),
                data={"customer": org["stripe_customer_id"],
                      "return_url": f"{FRONTEND_URL}/settings/billing"}
            )
            resp.raise_for_status()
            return {"success": True, "portal_url": resp.json()["url"]}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # Verify webhook signature
    if STRIPE_WEBHOOK_SECRET:
        try:
            timestamp = sig_header.split("t=")[1].split(",")[0]
            signatures = [s.split("v1=")[1] for s in sig_header.split(",") if s.startswith("v1=")]
            signed_payload = f"{timestamp}.{payload.decode()}"
            expected = hmac.new(STRIPE_WEBHOOK_SECRET.encode(), signed_payload.encode(), hashlib.sha256).hexdigest()
            if expected not in signatures:
                raise HTTPException(status_code=400, detail="Invalid webhook signature")
        except Exception as e:
            print(f"[webhook] Signature verification failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        event = json.loads(payload)
        event_type = event.get("type")
        print(f"[webhook] Received: {event_type}")

        # ── Payment succeeded → activate subscription ─────────
        if event_type in ("checkout.session.completed", "invoice.payment_succeeded"):
            obj = event["data"]["object"]
            org_id = obj.get("metadata", {}).get("org_id")
            plan   = obj.get("metadata", {}).get("plan")
            sub_id = obj.get("subscription") or obj.get("id")

            if org_id and plan:
                supabase.table("organisations").update({
                    "tier": plan, "updated_at": datetime.utcnow().isoformat()
                }).eq("id", org_id).execute()

                # Upsert subscription record
                supabase.table("subscriptions").upsert({
                    "org_id": org_id, "stripe_subscription_id": sub_id,
                    "plan": plan, "status": "active",
                    "updated_at": datetime.utcnow().isoformat(),
                }, on_conflict="stripe_subscription_id").execute()

                print(f"[webhook] Upgraded org {org_id} to {plan}")

        # ── Subscription cancelled → downgrade ────────────────
        elif event_type == "customer.subscription.deleted":
            obj = event["data"]["object"]
            org_id = obj.get("metadata", {}).get("org_id")
            sub_id = obj.get("id")
            if org_id:
                supabase.table("organisations").update({
                    "tier": "basic", "updated_at": datetime.utcnow().isoformat()
                }).eq("id", org_id).execute()
                supabase.table("subscriptions").update({
                    "status": "cancelled", "updated_at": datetime.utcnow().isoformat()
                }).eq("stripe_subscription_id", sub_id).execute()
                print(f"[webhook] Downgraded org {org_id} to basic")

        # ── Payment failed → mark past_due ────────────────────
        elif event_type == "invoice.payment_failed":
            obj = event["data"]["object"]
            sub_id = obj.get("subscription")
            if sub_id:
                supabase.table("subscriptions").update({
                    "status": "past_due", "updated_at": datetime.utcnow().isoformat()
                }).eq("stripe_subscription_id", sub_id).execute()
                print(f"[webhook] Payment failed for subscription {sub_id}")

        # ── Subscription updated (upgrade/downgrade) ──────────
        elif event_type == "customer.subscription.updated":
            obj = event["data"]["object"]
            org_id = obj.get("metadata", {}).get("org_id")
            plan   = obj.get("metadata", {}).get("plan")
            sub_id = obj.get("id")
            status = obj.get("status")
            if org_id and plan and status == "active":
                supabase.table("organisations").update({
                    "tier": plan, "updated_at": datetime.utcnow().isoformat()
                }).eq("id", org_id).execute()
                print(f"[webhook] Updated org {org_id} to {plan}")

        return {"received": True}

    except Exception as e:
        print(f"[webhook] Error: {e}")
        return {"received": True}  # Always return 200 to Stripe


@app.get("/billing/subscription")
def get_subscription(org_id: str = Query(...)):
    """Get current subscription details for an org."""
    try:
        result = (supabase.table("subscriptions").select("*")
                  .eq("org_id", org_id).eq("status", "active")
                  .order("created_at", desc=True).limit(1).execute())
        org = supabase.table("organisations").select("tier, trial_expires_at, stripe_customer_id").eq("id", org_id).single().execute().data
        return {"subscription": result.data[0] if result.data else None, "org": org}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ─── ADMIN ENDPOINTS ──────────────────────────────────────────

@app.post("/admin/expire-trials")
def expire_trials():
    """Expire trials that have passed their end date. Call daily via cron."""
    try:
        result = supabase.rpc("expire_trials").execute()
        return {"success": True, "expired": result.data}
    except Exception as e:
        return {"success": False, "message": str(e)}


# ─── SITES CRUD ───────────────────────────────────────────────

@app.get("/analytics/sites")
def get_sites():
    try:
        result = (supabase.table("sites").select("id,name,lat,lng,timezone,base_temp,mode,address,building_type")
                  .eq("is_active", True).order("name").execute())
        return {"sites": result.data or []}
    except Exception as e:
        return {"success": False, "message": str(e)}

@app.post("/analytics/sites")
def create_site(site: SiteCreate):
    try:
        base_temp = auto_base_temp(site.lat, site.base_temp)
        result = supabase.table("sites").insert({
            "name":site.name,"lat":site.lat,"lng":site.lng,"timezone":site.timezone,
            "base_temp":base_temp,"mode":site.mode,"address":site.address,
            "building_type":site.building_type,"is_active":True}).execute()
        return {"success":True,"site":result.data[0] if result.data else None}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.put("/analytics/sites/{site_id}")
def update_site(site_id: str, site: SiteCreate):
    try:
        result = supabase.table("sites").update({
            "name":site.name,"lat":site.lat,"lng":site.lng,"timezone":site.timezone,
            "base_temp":site.base_temp,"mode":site.mode,"address":site.address,
            "building_type":site.building_type}).eq("id",site_id).execute()
        return {"success":True,"site":result.data[0] if result.data else None}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.delete("/analytics/sites/{site_id}")
def delete_site(site_id: str):
    try:
        supabase.table("sites").update({"is_active":False}).eq("id",site_id).execute()
        return {"success":True,"deleted":site_id}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── WEATHER NORMALISED ───────────────────────────────────────

@app.get("/analytics/weather-normalised")
async def get_weather_normalised(
    site_id: str = Query(...), org_id: Optional[str] = Query(default=None),
    start_date: Optional[str] = Query(default=None), end_date: Optional[str] = Query(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_feature(org_id, "weather_normalisation")
    try:
        site = supabase.table("sites").select("*").eq("id",site_id).eq("is_active",True).single().execute().data
        if not site: return {"success":False,"message":f"Site not found"}
        base_temp=site["base_temp"]; mode=resolve_mode(site["mode"],site["lat"],base_temp)
        today=datetime.utcnow().date()
        if not end_date: end_date=str(today-timedelta(days=1))
        if not start_date: start_date=str(today-timedelta(days=31))
        data=(supabase.table("energy_data").select("timestamp,consumption")
              .gte("timestamp",start_date).lte("timestamp",end_date+"T23:59:59")
              .range(0,20000).execute().data)
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
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── ANOMALIES ────────────────────────────────────────────────

@app.get("/anomalies")
def get_anomalies(days: int=Query(default=90,ge=7,le=365),
                  severity: Optional[str]=Query(default=None),
                  anomaly_type: Optional[str]=Query(default=None)):
    try:
        end_dt=datetime.utcnow(); start_dt=end_dt-timedelta(days=days)
        data=(supabase.table("energy_data").select("timestamp,consumption")
              .gte("timestamp",start_dt.strftime("%Y-%m-%dT%H:%M:%S"))
              .lte("timestamp",end_dt.strftime("%Y-%m-%dT%H:%M:%S"))
              .range(0,20000).execute().data)
        if not data:
            return {"anomalies":[],"summary":{"total":0,"high":0,"medium":0,"low":0,"spikes":0,"drops":0},
                    "chartData":[],"avgDaily":0,"heatmap":[[0]*24 for _ in range(7)],"totalScanned":0}
        df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        df=df.sort_values("timestamp").reset_index(drop=True)
        df["date"]=df["timestamp"].dt.date.astype(str); df["hour"]=df["timestamp"].dt.hour
        df["dow"]=df["timestamp"].dt.dayofweek; df["hour_of_week"]=df["dow"]*24+df["hour"]
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
            anomalies.append({"timestamp":row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S"),
                "date":row["date"],"hour":int(row["hour"]),"hourLabel":f"{int(row['hour']):02d}:00",
                "actual":round(float(actual),2),"expected":round(float(expected),2),
                "deviationPct":round(float(dev_pct),1),"stdDeviations":round(float(std_dev),2),
                "severity":sev,"type":a_type,"dow":int(row["dow"]),
                "dowLabel":["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][int(row["dow"])]})
        filtered=list(anomalies)
        if severity: filtered=[a for a in filtered if a["severity"]==severity.lower()]
        if anomaly_type: filtered=[a for a in filtered if a["type"]==anomaly_type.lower()]
        sev_order={"high":0,"medium":1,"low":2}
        filtered.sort(key=lambda x:(sev_order[x["severity"]],x["timestamp"]))
        summary={"total":len(anomalies),"high":sum(1 for a in anomalies if a["severity"]=="high"),
                 "medium":sum(1 for a in anomalies if a["severity"]=="medium"),
                 "low":sum(1 for a in anomalies if a["severity"]=="low"),
                 "spikes":sum(1 for a in anomalies if a["type"]=="spike"),
                 "drops":sum(1 for a in anomalies if a["type"]=="drop")}
        daily=df.groupby("date")["consumption"].sum().reset_index()
        anomaly_dates={}
        for a in anomalies:
            d=a["date"]
            if d not in anomaly_dates or sev_order[a["severity"]]<sev_order[anomaly_dates[d]["sev"]]:
                anomaly_dates[d]={"sev":a["severity"]}
        chart_data=[{"date":str(row["date"]),"consumption":round(float(row["consumption"]),2),
                     "hasAnomaly":str(row["date"]) in anomaly_dates,
                     "anomalySev":anomaly_dates[str(row["date"])]["sev"] if str(row["date"]) in anomaly_dates else None,
                     "anomalyCount":sum(1 for a in anomalies if a["date"]==str(row["date"]))}
                    for _,row in daily.iterrows()]
        heatmap=[[0]*24 for _ in range(7)]
        for a in anomalies: heatmap[a["dow"]][a["hour"]]+=1
        print(f"[anomalies] {len(df)} rows → {len(anomalies)} anomalies")
        return {"anomalies":filtered,"summary":summary,"chartData":chart_data,
                "avgDaily":round(float(df.groupby("date")["consumption"].sum().mean()),2),
                "heatmap":heatmap,"totalScanned":len(df)}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── AI INSIGHTS ──────────────────────────────────────────────

async def generate_ai_insights_data(stats):
    elec,gas,comb=stats.get("electricity",{}),stats.get("gas",{}),stats.get("combined",{})
    prompt=f"""You are an expert energy analyst. Analyse this building energy data and return ONLY valid JSON.

ELECTRICITY: total={elec.get('total_kwh')}kWh cost=£{elec.get('total_cost_gbp')} avg_daily={elec.get('avg_daily_kwh')}kWh
baseload={elec.get('baseload_kwh')}kWh/h peak_hour={elec.get('peak_hour')}:00 off_hours={elec.get('off_hours_pct')}%
weekday={elec.get('avg_weekday_daily')}kWh weekend={elec.get('avg_weekend_daily')}kWh mom={elec.get('month_on_month_pct')}%
monthly={json.dumps(elec.get('monthly_breakdown',{}))} period={elec.get('data_from')} to {elec.get('data_to')}

GAS: total={gas.get('total_kwh')}kWh cost=£{gas.get('total_cost_gbp')} avg_daily={gas.get('avg_daily_kwh')}kWh

COMBINED: total={comb.get('total_energy_kwh')}kWh cost=£{comb.get('total_cost_gbp')}

Return ONLY this JSON:
{{"executive_summary":"2-3 sentences for business owner",
"insights":[{{"id":"slug","category":"baseload|peak_demand|off_hours|weekday_weekend|seasonal|gas|cost|trend",
"title":"Short title","finding":"2-3 sentences with numbers","implication":"Why this matters",
"severity":"high|medium|low|positive","audience":["facilities","consultant","executive"],
"metric":"key number","metric_label":"label"}}],
"recommendations":[{{"id":"slug","title":"Action title","action":"Specific step","rationale":"Why",
"saving_kwh_monthly":0,"saving_gbp_monthly":0,"effort":"low|medium|high",
"timeframe":"immediate|1_month|3_months|6_months","payback_months":0,
"category":"behavioural|controls|equipment|monitoring|procurement",
"audience":["facilities","consultant","executive"],"priority":"quick_win|medium_term|long_term"}}]}}

Generate 5-8 insights and 5-7 recommendations. Hotel context. Return ONLY JSON."""
    raw=await call_claude(prompt)
    raw=raw.strip()
    if raw.startswith("```"):
        raw=raw.split("```")[1]
        if raw.startswith("json"): raw=raw[4:]
    return json.loads(raw.strip())

async def run_ai_generation():
    print("[ai] Starting generation...")
    placeholder=supabase.table("ai_insights").insert({"status":"generating","generated_at":datetime.utcnow().isoformat()}).execute()
    row_id=placeholder.data[0]["id"] if placeholder.data else None
    try:
        stats=build_energy_summary_for_ai()
        if not stats.get("electricity") and not stats.get("gas"):
            raise ValueError("No energy data")
        result=await generate_ai_insights_data(stats)
        data_from=stats.get("electricity",{}).get("data_from") or stats.get("gas",{}).get("data_from")
        data_to=stats.get("electricity",{}).get("data_to") or stats.get("gas",{}).get("data_to")
        payload={"status":"complete","generated_at":datetime.utcnow().isoformat(),
                 "data_from":data_from,"data_to":data_to,
                 "executive_summary":result.get("executive_summary",""),
                 "insights":result.get("insights",[]),"recommendations":result.get("recommendations",[]),
                 "raw_stats":stats}
        if row_id: supabase.table("ai_insights").update(payload).eq("id",row_id).execute()
        else: supabase.table("ai_insights").insert(payload).execute()
        print(f"[ai] Done — {len(result.get('insights',[]))} insights")
    except Exception as e:
        print(f"[ai] Failed: {e}")
        if row_id: supabase.table("ai_insights").update({"status":"error","error_message":str(e)}).eq("id",row_id).execute()

@app.get("/ai/insights")
def get_ai_insights(org_id: Optional[str]=Query(default=None),
                    authorization: Optional[str]=Header(default=None)):
    require_feature(org_id, "ai_insights")
    try:
        result=(supabase.table("ai_insights").select("*").eq("status","complete")
                .order("generated_at",desc=True).limit(1).execute())
        if not result.data:
            pending=(supabase.table("ai_insights").select("id,status,generated_at")
                     .eq("status","generating").order("generated_at",desc=True).limit(1).execute())
            if pending.data: return {"status":"generating","insights":None}
            return {"status":"empty","insights":None}
        row=result.data[0]
        return {"status":"complete","generatedAt":row["generated_at"],"dataFrom":row["data_from"],
                "dataTo":row["data_to"],"executiveSummary":row["executive_summary"],
                "insights":row["insights"] or [],"recommendations":row["recommendations"] or []}
    except HTTPException: raise
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.post("/ai/insights/generate")
async def trigger_ai_generation(background_tasks: BackgroundTasks,
                                 org_id: Optional[str]=Query(default=None),
                                 authorization: Optional[str]=Header(default=None)):
    require_feature(org_id, "ai_insights")
    if not ANTHROPIC_API_KEY:
        return {"success":False,"message":"ANTHROPIC_API_KEY not configured"}
    background_tasks.add_task(run_ai_generation)
    return {"success":True,"message":"Generation started"}

@app.get("/ai/insights/history")
def get_ai_insights_history():
    try:
        result=(supabase.table("ai_insights").select("id,generated_at,data_from,data_to,status,error_message")
                .order("generated_at",desc=True).limit(10).execute())
        return {"history":result.data or []}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── AI ENERGY ANALYST ────────────────────────────────────────

@app.post("/ai/analyst/chat")
async def ai_analyst_chat(req: ChatRequest,
                           authorization: Optional[str]=Header(default=None)):
    require_feature(req.org_id, "ai_energy_analyst")
    try:
        stats=build_energy_summary_for_ai()
        elec=stats.get("electricity",{}); gas=stats.get("gas",{})
        system=f"""You are Enerlytics AI — expert energy analyst for this building.

ELECTRICITY: {elec.get('total_kwh','N/A')}kWh total £{elec.get('total_cost_gbp','N/A')} | Period: {elec.get('data_from')} to {elec.get('data_to')}
Avg daily: {elec.get('avg_daily_kwh')}kWh | Peak hour: {elec.get('peak_hour')}:00 | Baseload: {elec.get('baseload_kwh')}kWh/h
Off-hours: {elec.get('off_hours_pct')}% | Weekday: {elec.get('avg_weekday_daily')}kWh | Weekend: {elec.get('avg_weekend_daily')}kWh
MoM change: {elec.get('month_on_month_pct')}% | Monthly: {json.dumps(elec.get('monthly_breakdown',{}))}
GAS: {gas.get('total_kwh','N/A')}kWh £{gas.get('total_cost_gbp','N/A')}

Answer questions about this building's energy data specifically and general energy questions.
Be concise, use numbers from the data, format clearly with bullets for lists."""
        messages=[{"role":m.role,"content":m.content} for m in req.messages]
        response=await call_claude_messages(messages,system=system,max_tokens=2000)
        if req.org_id:
            all_msgs=messages+[{"role":"assistant","content":response}]
            if req.conversation_id:
                supabase.table("ai_conversations").update(
                    {"messages":all_msgs,"updated_at":datetime.utcnow().isoformat()}
                ).eq("id",req.conversation_id).execute()
            else:
                title=req.messages[0].content[:60]+("..." if len(req.messages[0].content)>60 else "")
                result=supabase.table("ai_conversations").insert(
                    {"org_id":req.org_id,"title":title,"messages":all_msgs}).execute()
                if result.data:
                    return {"response":response,"conversation_id":result.data[0]["id"]}
        return {"response":response,"conversation_id":req.conversation_id}
    except HTTPException: raise
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/ai/analyst/conversations")
def get_conversations(org_id: str=Query(...)):
    require_feature(org_id, "ai_energy_analyst")
    try:
        result=(supabase.table("ai_conversations").select("id,title,created_at,updated_at")
                .eq("org_id",org_id).order("updated_at",desc=True).limit(20).execute())
        return {"conversations":result.data or []}
    except HTTPException: raise
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/ai/analyst/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    try:
        result=supabase.table("ai_conversations").select("*").eq("id",conversation_id).single().execute()
        return {"conversation":result.data}
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── REPORTS ──────────────────────────────────────────────────

def build_report_data(date_from, date_to, report_type):
    elec_data=(supabase.table("energy_data").select("timestamp,consumption")
               .gte("timestamp",date_from).lte("timestamp",date_to+"T23:59:59")
               .range(0,20000).execute().data) or []
    gas_data=(supabase.table("gas_data").select("timestamp,consumption")
              .gte("timestamp",date_from).lte("timestamp",date_to+"T23:59:59")
              .range(0,20000).execute().data) or []
    report={"report_type":report_type,"date_from":date_from,"date_to":date_to,
            "generated_at":datetime.utcnow().isoformat()}
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
    ec=report.get("electricity",{}).get("total_cost_gbp",0)
    gc=report.get("gas",{}).get("total_cost_gbp",0)
    ek=report.get("electricity",{}).get("total_kwh",0)
    gk=report.get("gas",{}).get("total_kwh",0)
    report["combined"]={"total_kwh":round(ek+gk,2),"total_cost_gbp":round(ec+gc,2),
        "electricity_share_pct":round(ek/(ek+gk)*100,1) if (ek+gk) else 0,
        "gas_share_pct":round(gk/(ek+gk)*100,1) if (ek+gk) else 0}
    if report_type in ("ai_insights","full","premium_full"):
        ai_result=(supabase.table("ai_insights").select("executive_summary,insights,recommendations")
                   .eq("status","complete").order("generated_at",desc=True).limit(1).execute())
        if ai_result.data:
            row=ai_result.data[0]
            report["ai_insights"]={"executive_summary":row["executive_summary"],
                "insights":row["insights"] or [],"recommendations":row["recommendations"] or []}
    return report

@app.post("/reports/generate")
def generate_report(req: ReportRequest):
    feature_map={"basic":"report_basic","ai_insights":"report_ai_insights",
                 "full":"report_full","premium_full":"report_premium_full"}
    require_feature(req.org_id, feature_map.get(req.report_type,"report_basic"))
    try:
        data=build_report_data(req.date_from,req.date_to,req.report_type)
        titles={"basic":"Basic Energy Report","ai_insights":"AI Insights Report",
                "full":"Full Energy Report","premium_full":"Premium Full Report"}
        title=f"{titles.get(req.report_type,'Report')} — {req.date_from} to {req.date_to}"
        result=supabase.table("reports").insert({"org_id":req.org_id,"report_type":req.report_type,
            "title":title,"date_from":req.date_from,"date_to":req.date_to,
            "period_type":req.period_type,"status":"complete","data":data}).execute()
        return {"success":True,"report_id":result.data[0]["id"] if result.data else None,
                "title":title,"data":data}
    except HTTPException: raise
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/reports")
def list_reports(org_id: Optional[str]=Query(default=None)):
    try:
        query=supabase.table("reports").select("id,report_type,title,date_from,date_to,period_type,status,created_at")
        if org_id: query=query.eq("org_id",org_id)
        result=query.order("created_at",desc=True).limit(50).execute()
        return {"reports":result.data or []}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/reports/{report_id}")
def get_report(report_id: str):
    try:
        result=supabase.table("reports").select("*").eq("id",report_id).single().execute()
        if not result.data: return {"success":False,"message":"Report not found"}
        return {"report":result.data}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.delete("/reports/{report_id}")
def delete_report(report_id: str):
    try:
        supabase.table("reports").delete().eq("id",report_id).execute()
        return {"success":True,"deleted":report_id}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/reports/preview/{report_type}")
def preview_report(report_type: str, date_from: str=Query(...), date_to: str=Query(...),
                   org_id: Optional[str]=Query(default=None)):
    feature_map={"basic":"report_basic","ai_insights":"report_ai_insights",
                 "full":"report_full","premium_full":"report_premium_full"}
    require_feature(org_id, feature_map.get(report_type,"report_basic"))
    try:
        return {"success":True,"data":build_report_data(date_from,date_to,report_type)}
    except HTTPException: raise
    except Exception as e:
        return {"success":False,"message":str(e)}

# ─── UPLOAD / ANALYTICS / GAS / DEBUG / DELETE ────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile=File(...)):
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
        for i in range(0,len(records),500): supabase.table("energy_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Data stored"}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/analytics")
def get_analytics():
    data=supabase.table("energy_data").select("*").range(0,20000).execute().data
    if not data:
        return {"stats":{"baseload":0,"peakDemand":0,"loadFactor":0,"avgDaily":0},
                "hourlyProfile":[],"daily":[],"totalConsumption":0,"heatmap":[[0.0]*24 for _ in range(7)]}
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
    df["hour"]=df["timestamp"].dt.hour; df["is_weekend"]=df["timestamp"].dt.dayofweek>=5
    df["date"]=df["timestamp"].dt.date
    stats=build_stats(df); total=round(float(df["consumption"].sum()),2)
    daily=df.groupby("date")["consumption"].sum().reset_index()
    daily_breakdown=[]
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
def get_hourly_profile_by_year(year: int):
    try:
        data=(supabase.table("energy_data").select("timestamp,consumption")
              .gte("timestamp",f"{year}-01-01").lte("timestamp",f"{year}-12-31T23:59:59").execute().data)
        if not data:
            return {"hourlyProfile":[{"hour":f"{h:02d}:00","average":0,"weekday":0,"weekend":0} for h in range(24)]}
        df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        return {"hourlyProfile":build_hourly_profile(df)}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.post("/upload-gas-data")
async def upload_gas_data(file: UploadFile=File(...)):
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
        for i in range(0,len(records),500): supabase.table("gas_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Gas data stored"}
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/gas-analytics")
def get_gas_analytics():
    data=supabase.table("gas_data").select("*").range(0,20000).execute().data
    if not data:
        return {"stats":{"baseload":0,"peakDemand":0,"loadFactor":0,"avgDaily":0},"hourlyProfile":[],"daily":[],"totalConsumption":0}
    df=pd.DataFrame(data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
    df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df); df["date"]=df["timestamp"].dt.date
    daily=df.groupby("date")["consumption"].sum().reset_index()
    return {"stats":build_stats(df),"hourlyProfile":build_hourly_profile(df),
            "daily":[{"date":str(r["date"]),"consumption":round(float(r["consumption"]),2)} for _,r in daily.iterrows()],
            "totalConsumption":round(float(df["consumption"].sum()),2)}

@app.get("/debug/data-summary")
def debug_data_summary():
    try:
        data=supabase.table("energy_data").select("*").range(0,20000).execute().data
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
    except Exception as e:
        return {"success":False,"message":str(e)}

@app.get("/debug/gas-summary")
def debug_gas_summary():
    try:
        data=supabase.table("gas_data").select("*").range(0,20000).execute().data
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
    except Exception as e:
        return {"success":False,"message":str(e)}

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
