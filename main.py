@app.get("/analytics")
def analytics():
    return {
        "stats": {
            "totalConsumption": 12000,
            "avgDaily": 400,
            "peakDemand": 250,
            "peakDay": "Monday",
            "estimatedCost": 1500,
            "baseload": 120,
            "daysOfData": 30,
            "trend": {
                "consumptionChange": 5,
                "costChange": 3
            }
        },
        "daily": [
            {"date": "2024-01-01", "label": "Jan 1", "consumption": 400}
        ],
        "weekly": [
            {"week": "Week 1", "consumption": 2800}
        ],
        "monthly": [
            {"month": "Jan", "consumption": 12000}
        ]
    }
