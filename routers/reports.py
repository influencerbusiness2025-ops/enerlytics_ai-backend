"""
reports.py — Energy Report endpoints for Effictra AI
Mounted in main.py via: app.include_router(reports_router)
"""
from fastapi import APIRouter, Header, Query
from typing import Optional
from .dependencies import (
    supabase, supabase_service,
    ELECTRICITY_RATE_GBP, GAS_RATE_GBP,
    ANTHROPIC_MODEL, ANTHROPIC_API_KEY, FRONTEND_URL,
    require_feature_jwt, parse_timestamps_naive, build_hourly_profile,
    get_period_date_range, ReportRequest,
)
import pandas as pd
from datetime import datetime

router = APIRouter(prefix="/reports", tags=["reports"])

def build_report_data(date_from,date_to,report_type,org_id=None,site_id=None):
    elec_q=supabase.table("energy_data").select("timestamp,consumption").gte("timestamp",date_from).lte("timestamp",date_to+"T23:59:59").range(0,20000)
    gas_q=supabase.table("gas_data").select("timestamp,consumption").gte("timestamp",date_from).lte("timestamp",date_to+"T23:59:59").range(0,20000)
    if site_id: elec_q=elec_q.eq("site_id",site_id); gas_q=gas_q.eq("site_id",site_id)
    elif org_id: elec_q=elec_q.eq("org_id",org_id); gas_q=gas_q.eq("org_id",org_id)
    elec_data=elec_q.execute().data or []
    gas_data=gas_q.execute().data or []
    # Clamp date_to to actual latest data to avoid phantom future months
    actual_date_to = date_to
    if elec_data:
        latest_ts = max(r["timestamp"] for r in elec_data)
        actual_date_to = latest_ts[:10]
    report={"report_type":report_type,"date_from":date_from,"date_to":actual_date_to,"generated_at":datetime.utcnow().isoformat()}
    if elec_data:
        df=pd.DataFrame(elec_data); df["consumption"]=pd.to_numeric(df["consumption"],errors="coerce")
        df=df.dropna(subset=["consumption"]); df=parse_timestamps_naive(df)
        df["date"]=df["timestamp"].dt.date.astype(str)
        # Use year-month string to avoid month-number collisions across years
        df["year_month"]=df["timestamp"].dt.strftime("%Y-%m")
        daily=df.groupby("date")["consumption"].sum()
        monthly=df.groupby("year_month")["consumption"].sum()
        total_kwh=round(float(df["consumption"].sum()),2)
        days_of_data=len(daily)
        report["electricity"]={"total_kwh":total_kwh,"total_cost_gbp":round(total_kwh*ELECTRICITY_RATE_GBP,2),
            "avg_daily_kwh":round(float(daily.mean()),2),"peak_daily_kwh":round(float(daily.max()),2),
            "days_of_data":days_of_data,
            "baseload_kwh":round(float(df["consumption"].quantile(0.1)),2),
            "peak_demand_kwh":round(float(df["consumption"].max()),2),
            "hourly_profile":build_hourly_profile(df),
            "daily_breakdown":[{"date":d,"kwh":round(float(v),2),"cost":round(float(v)*ELECTRICITY_RATE_GBP,2)} for d,v in daily.items()],
            "monthly_breakdown":[{"month":m,"kwh":round(float(v),2),"cost":round(float(v)*ELECTRICITY_RATE_GBP,2)} for m,v in monthly.items()]}
    if gas_data:
        gdf=pd.DataFrame(gas_data); gdf["consumption"]=pd.to_numeric(gdf["consumption"],errors="coerce")
        gdf=gdf.dropna(subset=["consumption"]); gdf=parse_timestamps_naive(gdf)
        gdf["date"]=gdf["timestamp"].dt.date.astype(str)
        gdf["year_month"]=gdf["timestamp"].dt.strftime("%Y-%m")
        gas_daily=gdf.groupby("date")["consumption"].sum()
        gas_monthly=gdf.groupby("year_month")["consumption"].sum()
        gas_total=round(float(gdf["consumption"].sum()),2)
        report["gas"]={"total_kwh":gas_total,"total_cost_gbp":round(gas_total*GAS_RATE_GBP,2),
            "avg_daily_kwh":round(float(gas_daily.mean()),2),
            "daily_breakdown":[{"date":d,"kwh":round(float(v),2),"cost":round(float(v)*GAS_RATE_GBP,2)} for d,v in gas_daily.items()],
            "monthly_breakdown":[{"month":m,"kwh":round(float(v),2),"cost":round(float(v)*GAS_RATE_GBP,2)} for m,v in gas_monthly.items()]}
    ek=report.get("electricity",{}).get("total_kwh",0); gk=report.get("gas",{}).get("total_kwh",0)
    ec=report.get("electricity",{}).get("total_cost_gbp",0); gc=report.get("gas",{}).get("total_cost_gbp",0)
    report["combined"]={"total_kwh":round(ek+gk,2),"total_cost_gbp":round(ec+gc,2),
        "electricity_share_pct":round(ek/(ek+gk)*100,1) if (ek+gk) else 0,
        "gas_share_pct":round(gk/(ek+gk)*100,1) if (ek+gk) else 0}
    if report_type in ("ai_insights","full","premium_full"):
        ai_q=supabase.table("ai_insights").select("executive_summary,insights,recommendations").eq("status","complete").order("generated_at",desc=True)
        if site_id: ai_q=ai_q.eq("site_id",site_id)
        elif org_id: ai_q=ai_q.eq("org_id",org_id)
        ai_result=ai_q.limit(1).execute()
        if ai_result.data:
            row=ai_result.data[0]
            report["ai_insights"]={"executive_summary":row["executive_summary"],
                "insights":row["insights"] or [],"recommendations":row["recommendations"] or []}
    return report

@router.post("/generate")
def generate_report(req: ReportRequest, authorization: Optional[str]=Header(default=None)):
    feature_map={"basic":"report_basic","ai_insights":"report_ai_insights","full":"report_full","premium_full":"report_premium_full"}
    require_feature_jwt(authorization,req.org_id,feature_map.get(req.report_type,"report_basic"))
    try:
        data=build_report_data(req.date_from,req.date_to,req.report_type,org_id=req.org_id,site_id=getattr(req,"site_id",None))
        titles={"basic":"Basic Energy Report","ai_insights":"AI Insights Report","full":"Full Energy Report","premium_full":"Premium Full Report"}
        title=f"{titles.get(req.report_type,'Report')} — {req.date_from} to {req.date_to}"
        result=supabase.table("reports").insert({"org_id":req.org_id,"report_type":req.report_type,"title":title,
            "date_from":req.date_from,"date_to":req.date_to,"period_type":req.period_type,"status":"complete","data":data}).execute()
        return {"success":True,"report_id":result.data[0]["id"] if result.data else None,"title":title,"data":data}
    except HTTPException: raise
    except Exception as e: return {"success":False,"message":str(e)}

@router.get("")
def list_reports(org_id: Optional[str]=Query(default=None)):
    try:
        query=supabase.table("reports").select("id,report_type,title,date_from,date_to,period_type,status,created_at")
        if org_id: query=query.eq("org_id",org_id)
        result=query.order("created_at",desc=True).limit(50).execute()
        return {"reports":result.data or []}
    except Exception as e: return {"success":False,"message":str(e)}

@router.get("/{report_id}")
def get_report(report_id: str):
    try:
        result=supabase.table("reports").select("*").eq("id",report_id).single().execute()
        if not result.data: return {"success":False,"message":"Report not found"}
        return {"report":result.data}
    except Exception as e: return {"success":False,"message":str(e)}

@router.delete("/{report_id}")
def delete_report(report_id: str):
    try: supabase.table("reports").delete().eq("id",report_id).execute(); return {"success":True,"deleted":report_id}
    except Exception as e: return {"success":False,"message":str(e)}

@router.get("/preview/{report_type}")
def preview_report(report_type: str,date_from: str=Query(...),date_to: str=Query(...),
                   org_id: Optional[str]=Query(default=None),site_id: Optional[str]=Query(default=None),
                   authorization: Optional[str]=Header(default=None)):
    feature_map={"basic":"report_basic","ai_insights":"report_ai_insights","full":"report_full","premium_full":"report_premium_full"}
    require_feature_jwt(authorization,org_id,feature_map.get(report_type,"report_basic"))
    try: return {"success":True,"data":build_report_data(date_from,date_to,report_type,org_id=org_id,site_id=site_id)}
    except HTTPException: raise
    except Exception as e: return {"success":False,"message":str(e)}
