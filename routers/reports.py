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
    get_period_date_range,
    ReportRequest,
)
import pandas as pd
from datetime import datetime

router = APIRouter(prefix="/reports", tags=["reports"])

# Re-expose router endpoints without /reports prefix for preview (legacy)
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

@app.get("/reports")
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

# ─── UPLOAD / ANALYTICS ───────────────────────────────────────

@app.post("/upload-data")
async def upload_data(file: UploadFile=File(...), org_id: Optional[str]=Query(default=None),
                      site_id: Optional[str]=Query(default=None),
                      authorization: Optional[str]=Header(default=None)):
    try:
        contents=await file.read(); df=pd.read_csv(StringIO(contents.decode("utf-8")),index_col=None)
        date_col=next((c for c in df.columns if "date" in c.lower()),None)
        if not date_col: return {"success":False,"message":"No date column"}
        time_columns=[col for col in df.columns if ":" in col]
        if not time_columns: return {"success":False,"message":"No time columns"}
        df_long=df.melt(id_vars=[date_col],value_vars=time_columns,var_name="time",value_name="consumption")
        df_long["consumption"]=pd.to_numeric(df_long["consumption"],errors="coerce")
        df_long=df_long.dropna(subset=["consumption"])
        # Handle 24:00:00 (end-of-day) → roll to 00:00:00 next day
        def parse_halfhourly_timestamp(row):
            time_str = str(row["time"]).strip()
            date_str = str(row[date_col]).strip()
            if time_str == "24:00:00" or time_str == "24:00":
                try:
                    return pd.to_datetime(date_str, dayfirst=True, errors="coerce") + pd.Timedelta(days=1)
                except: return pd.NaT
            return pd.to_datetime(date_str + " " + time_str, dayfirst=True, errors="coerce")
        df_long["timestamp"] = df_long.apply(parse_halfhourly_timestamp, axis=1)
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
        for r in records:
            if resolved_org_id: r["org_id"] = resolved_org_id
            if site_id: r["site_id"] = site_id
        for i in range(0,len(records),500): supabase.table("energy_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Data stored"}
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/analytics")
def get_analytics(org_id: Optional[str]=Query(default=None),
                  site_id: Optional[str]=Query(default=None),
                  start_date: Optional[str]=Query(default=None),
                  end_date: Optional[str]=Query(default=None),
                  authorization: Optional[str]=Header(default=None)):
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    q=supabase.table("energy_data").select("*").range(0,20000)
    if site_id: q=q.eq("site_id",site_id)
    elif resolved_org_id: q=q.eq("org_id",resolved_org_id)
    if start_date: q=q.gte("timestamp",start_date)
    if end_date: q=q.lte("timestamp",end_date+"T23:59:59")
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
async def upload_gas_data(file: UploadFile=File(...), org_id: Optional[str]=Query(default=None),
                           site_id: Optional[str]=Query(default=None),
                           authorization: Optional[str]=Header(default=None)):
    try:
        contents=await file.read(); df=pd.read_csv(StringIO(contents.decode("utf-8")),index_col=None)
        date_col=next((c for c in df.columns if "date" in c.lower()),None)
        if not date_col: return {"success":False,"message":"No date column"}
        time_columns=[col for col in df.columns if ":" in col]
        if not time_columns: return {"success":False,"message":"No time columns"}
        df_long=df.melt(id_vars=[date_col],value_vars=time_columns,var_name="time",value_name="consumption")
        df_long["consumption"]=pd.to_numeric(df_long["consumption"],errors="coerce")
        df_long=df_long.dropna(subset=["consumption"])
        # Handle 24:00:00 (end-of-day) → roll to 00:00:00 next day
        def parse_halfhourly_timestamp_gas(row):
            time_str = str(row["time"]).strip()
            date_str = str(row[date_col]).strip()
            if time_str == "24:00:00" or time_str == "24:00":
                try:
                    return pd.to_datetime(date_str, dayfirst=True, errors="coerce") + pd.Timedelta(days=1)
                except: return pd.NaT
            return pd.to_datetime(date_str + " " + time_str, dayfirst=True, errors="coerce")
        df_long["timestamp"] = df_long.apply(parse_halfhourly_timestamp_gas, axis=1)
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
        for r in records:
            if resolved_org_id: r["org_id"] = resolved_org_id
            if site_id: r["site_id"] = site_id
        for i in range(0,len(records),500): supabase.table("gas_data").insert(records[i:i+500]).execute()
        return {"success":True,"rowsProcessed":len(records),"message":"Gas data stored as kWh"}
    except Exception as e: return {"success":False,"message":str(e)}

@app.get("/gas-analytics")
def get_gas_analytics(org_id: Optional[str]=Query(default=None),
                      site_id: Optional[str]=Query(default=None),
                      start_date: Optional[str]=Query(default=None),
                      end_date: Optional[str]=Query(default=None),
                      authorization: Optional[str]=Header(default=None)):
    resolved_org_id = org_id
    if not resolved_org_id and authorization:
        try:
            _, org = require_auth(authorization)
            if org: resolved_org_id = org.get("id")
        except: pass
    q=supabase.table("gas_data").select("*").range(0,20000)
    if site_id: q=q.eq("site_id",site_id)
    elif resolved_org_id: q=q.eq("org_id",resolved_org_id)
    if start_date: q=q.gte("timestamp",start_date)
    if end_date: q=q.lte("timestamp",end_date+"T23:59:59")
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
def delete_data(site_id: Optional[str]=Query(default=None), org_id: Optional[str]=Query(default=None)):
    try:
        q = supabase.table("energy_data").delete()
        if site_id: q = q.eq("site_id", site_id)
        elif org_id: q = q.eq("org_id", org_id)
        else: q = q.gt("id", "00000000-0000-0000-0000-000000000000")
        q.execute()
        return {"success": True, "message": "Energy data deleted"}
    except Exception as e: return {"success": False, "message": str(e)}

@app.delete("/delete-gas-data")
def delete_gas_data(site_id: Optional[str]=Query(default=None), org_id: Optional[str]=Query(default=None)):
    try:
        q = supabase.table("gas_data").delete()
        if site_id: q = q.eq("site_id", site_id)
        elif org_id: q = q.eq("org_id", org_id)
        else: q = q.gt("id", "00000000-0000-0000-0000-000000000000")
        q.execute()
        return {"success": True, "message": "Gas data deleted"}
    except Exception as e: return {"success": False, "message": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
