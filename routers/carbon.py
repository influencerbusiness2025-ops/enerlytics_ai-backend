"""
carbon.py — Carbon Report endpoints for Effictra AI
Mounted in main.py via: app.include_router(carbon_router)
"""
from fastapi import APIRouter, Header, Query, Depends
from fastapi.responses import Response as _Response
from typing import Optional, List
from .dependencies import (
    supabase, supabase_service,
    ELECTRICITY_RATE_GBP, GAS_RATE_GBP,
    ANTHROPIC_MODEL, ANTHROPIC_API_KEY,
    require_feature_jwt, require_auth,
)
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import csv, io

router = APIRouter(prefix="/carbon", tags=["carbon"])

# CARBON REPORT ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
# UK DESNZ 2024 emission factors (kgCO2e per kWh)
SCOPE1_GAS_FACTOR     = 0.18316   # Natural gas combustion
SCOPE2_ELEC_LOCATION  = 0.20707   # Grid electricity, location-based
SCOPE2_ELEC_MARKET    = 0.19338   # Grid electricity, market-based (renewable tariff)

import calendar as _calendar
from fastapi.responses import Response as _Response

class CarbonMonth(BaseModel):
    month: str
    month_label: str
    electricity_kwh: float
    gas_kwh: float
    scope1_tco2e: float
    scope2_tco2e: float
    total_tco2e: float

class CarbonReport(BaseModel):
    period_label: str
    scope1_total: float
    scope2_total: float
    total_tco2e: float
    carbon_intensity: float
    scope1_pct: float
    scope2_pct: float
    prior_period_change_pct: Optional[float]
    months: List[CarbonMonth]
    ai_narrative: Optional[str]
    emission_factors: dict
    generated_at: str


def _require_carbon(authorization: Optional[str] = Header(default=None)):
    """Auth + carbon_reporting feature check."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization required")
    try:
        user, org = require_auth(authorization)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
    tier = (org or {}).get("tier", "basic")
    features = TIER_FEATURES.get(tier, TIER_FEATURES["basic"])
    if not features.get("carbon_reporting", False):
        raise HTTPException(status_code=403, detail="carbon_reporting feature requires Standard plan or above")
    org_id = (org or {}).get("id")
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id not found")
    return org_id


@router.get("/report", response_model=CarbonReport)
def get_carbon_report(
    period: str = Query("12m", description="12m | 6m | 3m | ytd | custom"),
    date_from: Optional[str] = Query(None),
    date_to:   Optional[str] = Query(None),
    include_ai: bool = Query(True),
    org_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(default=None),
):
    # ── resolve org_id ──────────────────────────────────────────
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org:
                resolved_org_id = org.get("id")
                tier = org.get("tier", "basic")
                if not TIER_FEATURES.get(tier, {}).get("carbon_reporting", False):
                    raise HTTPException(status_code=403, detail="carbon_reporting requires Standard plan or above")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    if not resolved_org_id:
        raise HTTPException(status_code=400, detail="org_id required")

    # ── date range ───────────────────────────────────────────────
    now = datetime.utcnow()
    if period == "6m":
        days = 180
    elif period == "3m":
        days = 90
    elif period == "ytd":
        days = (now - now.replace(month=1, day=1)).days or 1
    elif period == "custom" and date_from and date_to:
        start_dt = datetime.fromisoformat(date_from)
        end_dt   = datetime.fromisoformat(date_to)
        days     = max((end_dt - start_dt).days, 1)
    else:
        days = 365

    if period != "custom" or not (date_from and date_to):
        end_dt   = now
        start_dt = now - timedelta(days=days)

    prior_end   = start_dt
    prior_start = start_dt - timedelta(days=days)

    # ── fetch data ───────────────────────────────────────────────
    def fetch_monthly(table: str):
        rows = supabase.table(table) \
            .select("timestamp,consumption") \
            .eq("org_id", resolved_org_id) \
            .gte("timestamp", start_dt.isoformat()) \
            .lte("timestamp", end_dt.isoformat()) \
            .execute().data or []
        monthly: dict = {}
        for r in rows:
            try:
                key = str(r["timestamp"])[:7]   # "YYYY-MM"
                monthly[key] = monthly.get(key, 0.0) + float(r["consumption"] or 0)
            except Exception:
                pass
        return monthly

    monthly_elec = fetch_monthly("energy_data")
    monthly_gas  = fetch_monthly("gas_data")

    # ── build monthly breakdown ───────────────────────────────────
    all_months = sorted(set(list(monthly_elec.keys()) + list(monthly_gas.keys())))
    months_out: List[CarbonMonth] = []
    for m in all_months:
        e_kwh = monthly_elec.get(m, 0.0)
        g_kwh = monthly_gas.get(m, 0.0)
        s1    = round((g_kwh * SCOPE1_GAS_FACTOR)    / 1000, 3)
        s2    = round((e_kwh * SCOPE2_ELEC_LOCATION) / 1000, 3)
        yr, mo = int(m[:4]), int(m[5:7])
        months_out.append(CarbonMonth(
            month=m,
            month_label=f"{_calendar.month_abbr[mo]} {yr}",
            electricity_kwh=round(e_kwh, 1),
            gas_kwh=round(g_kwh, 1),
            scope1_tco2e=s1,
            scope2_tco2e=s2,
            total_tco2e=round(s1 + s2, 3),
        ))

    # ── totals ────────────────────────────────────────────────────
    total_elec    = sum(monthly_elec.values())
    total_gas     = sum(monthly_gas.values())
    scope1_total  = round((total_gas  * SCOPE1_GAS_FACTOR)    / 1000, 2)
    scope2_total  = round((total_elec * SCOPE2_ELEC_LOCATION) / 1000, 2)
    grand_total   = round(scope1_total + scope2_total, 2)
    total_kwh     = total_elec + total_gas
    intensity     = round((grand_total * 1000) / total_kwh, 5) if total_kwh > 0 else 0.0

    # ── prior period comparison ───────────────────────────────────
    prior_change: Optional[float] = None
    try:
        def sum_kwh(table: str):
            rows = supabase.table(table) \
                .select("consumption") \
                .eq("org_id", resolved_org_id) \
                .gte("timestamp", prior_start.isoformat()) \
                .lte("timestamp", prior_end.isoformat()) \
                .execute().data or []
            return sum(float(r["consumption"] or 0) for r in rows)

        prior_elec_kwh = sum_kwh("energy_data")
        prior_gas_kwh  = sum_kwh("gas_data")
        prior_total    = ((prior_gas_kwh  * SCOPE1_GAS_FACTOR)    / 1000 +
                          (prior_elec_kwh * SCOPE2_ELEC_LOCATION) / 1000)
        if prior_total > 0:
            prior_change = round(((grand_total - prior_total) / prior_total) * 100, 1)
    except Exception:
        pass

    # ── AI narrative ──────────────────────────────────────────────
    ai_narrative: Optional[str] = None
    if include_ai and grand_total > 0 and ANTHROPIC_API_KEY:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            summary = (
                f"Carbon report for period {start_dt.date()} to {end_dt.date()}.\n"
                f"Total: {grand_total} tCO2e. "
                f"Scope 1 (gas): {scope1_total} tCO2e ({round(scope1_total/grand_total*100,1) if grand_total else 0}%). "
                f"Scope 2 (electricity): {scope2_total} tCO2e ({round(scope2_total/grand_total*100,1) if grand_total else 0}%). "
                f"Total electricity: {round(total_elec)} kWh. Total gas: {round(total_gas)} kWh. "
                f"Carbon intensity: {intensity} kgCO2e/kWh. "
                f"Change vs prior period: {prior_change}%.\n"
                f"Recent monthly data (last 6 months): "
                f"{[{'month': m.month_label, 'scope1': m.scope1_tco2e, 'scope2': m.scope2_tco2e} for m in months_out[-6:]]}"
            )
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": (
                        "You are an energy and carbon consultant. Write a 2-3 sentence insight for this carbon report. "
                        "Be specific about the numbers. Highlight the dominant emission source and suggest one practical "
                        "reduction measure with an estimated impact. Keep it concise and actionable.\n\n" + summary
                    )
                }]
            )
            ai_narrative = resp.content[0].text.strip()
        except Exception:
            pass

    return CarbonReport(
        period_label=f"{start_dt.strftime('%b %Y')} – {end_dt.strftime('%b %Y')}",
        scope1_total=scope1_total,
        scope2_total=scope2_total,
        total_tco2e=grand_total,
        carbon_intensity=intensity,
        scope1_pct=round(scope1_total / grand_total * 100, 1) if grand_total else 0.0,
        scope2_pct=round(scope2_total / grand_total * 100, 1) if grand_total else 0.0,
        prior_period_change_pct=prior_change,
        months=months_out,
        ai_narrative=ai_narrative,
        emission_factors={
            "scope1_gas":            SCOPE1_GAS_FACTOR,
            "scope2_elec_location":  SCOPE2_ELEC_LOCATION,
            "scope2_elec_market":    SCOPE2_ELEC_MARKET,
            "standard":              "DESNZ 2024",
        },
        generated_at=datetime.utcnow().isoformat(),
    )


@router.get("/export/csv")
def export_carbon_csv(
    period: str = Query("12m"),
    date_from: Optional[str] = Query(None),
    date_to:   Optional[str] = Query(None),
    org_id: Optional[str] = Query(None),
    authorization: Optional[str] = Header(default=None),
):
    # ── resolve org_id ──────────────────────────────────────────
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org:
                resolved_org_id = org.get("id")
                tier = org.get("tier", "basic")
                if not TIER_FEATURES.get(tier, {}).get("carbon_reporting", False):
                    raise HTTPException(status_code=403, detail="carbon_reporting requires Standard plan or above")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    if not resolved_org_id:
        raise HTTPException(status_code=400, detail="org_id required")

    now = datetime.utcnow()
    if period == "6m":
        days = 180
    elif period == "3m":
        days = 90
    elif period == "custom" and date_from and date_to:
        start_dt = datetime.fromisoformat(date_from)
        end_dt   = datetime.fromisoformat(date_to)
        days     = max((end_dt - start_dt).days, 1)
    else:
        days = 365

    if period != "custom" or not (date_from and date_to):
        end_dt   = now
        start_dt = now - timedelta(days=days)

    def fetch_monthly(table: str):
        rows = supabase.table(table) \
            .select("timestamp,consumption") \
            .eq("org_id", resolved_org_id) \
            .gte("timestamp", start_dt.isoformat()) \
            .lte("timestamp", end_dt.isoformat()) \
            .execute().data or []
        monthly: dict = {}
        for r in rows:
            try:
                key = str(r["timestamp"])[:7]
                monthly[key] = monthly.get(key, 0.0) + float(r["consumption"] or 0)
            except Exception:
                pass
        return monthly

    monthly_elec = fetch_monthly("energy_data")
    monthly_gas  = fetch_monthly("gas_data")
    all_months   = sorted(set(list(monthly_elec.keys()) + list(monthly_gas.keys())))

    lines = [
        "Month,Electricity (kWh),Gas (kWh),Scope 1 tCO2e (Gas),Scope 2 tCO2e (Electricity),Total tCO2e",
        f"# Emission factors: Scope 1 gas = {SCOPE1_GAS_FACTOR} kgCO2e/kWh  |  Scope 2 electricity = {SCOPE2_ELEC_LOCATION} kgCO2e/kWh (DESNZ 2024 location-based)",
    ]
    for m in all_months:
        e = monthly_elec.get(m, 0.0)
        g = monthly_gas.get(m, 0.0)
        s1 = round((g * SCOPE1_GAS_FACTOR)    / 1000, 3)
        s2 = round((e * SCOPE2_ELEC_LOCATION) / 1000, 3)
        lines.append(f"{m},{round(e,1)},{round(g,1)},{s1},{s2},{round(s1+s2,3)}")

    csv_content = "\n".join(lines)
    return _Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=carbon_report_{period}.csv"},
    )
