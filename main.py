from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Enerlytics API running 🚀"}

@app.get("/analytics")
def analytics():
    return {
        "total_consumption": 12000,
        "peak_demand": 250,
        "average_daily": 400
    }
