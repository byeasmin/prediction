import os
import base64
import httpx
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# Create the FastAPI app instance
app = FastAPI(
    title="AI-Powered Environmental Intelligence API",
    description="An API that combines NASA POWER climate data with Gemini AI to generate human-readable environmental insights (text + images).",
)


# Models

class NASAData(BaseModel):
    T2M: dict = Field(..., description="Daily average temperature at 2 meters (°C).")
    PRECTOT: dict = Field(..., description="Daily precipitation (mm/day).")
    WS2M: dict = Field(..., description="Wind speed at 2 meters (m/s).")
    RH2M: dict = Field(..., description="Relative humidity at 2 meters (%).")

class AIResponse(BaseModel):
    request_latitude: float
    request_longitude: float
    nasa_data_used: dict
    user_query: str
    ai_environmental_analysis: str


# Helper functions

async def fetch_nasa_power_data(lat: float, lon: float, start: date, end: date) -> dict:
    power_api_url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,PRECTOT,WS2M,RH2M"
        f"&community=RE"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start.strftime('%Y%m%d')}"
        f"&end={end.strftime('%Y%m%d')}"
        "&format=JSON"
    )

    async with httpx.AsyncClient() as client:
        response = await client.get(power_api_url, timeout=15.0)
        response.raise_for_status()
        nasa_data = response.json()

    parameters = nasa_data.get("properties", {}).get("parameter", {})
    if not parameters:
        raise HTTPException(status_code=404, detail="NASA POWER data not found for this location/date range.")

    return {
        "T2M": parameters.get("T2M", {}),
        "PRECTOT": parameters.get("PRECTOT", {}),
        "WS2M": parameters.get("WS2M", {}),
        "RH2M": parameters.get("RH2M", {}),
    }


async def call_gemini_api(prompt_text: str, image_base64: str = None) -> str:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in environment variables.")

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}

    if image_base64:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }
    else:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text}
                    ]
                }
            ]
        }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{gemini_url}?key={gemini_api_key}",
            headers=headers,
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        gemini_data = response.json()

    return (
        gemini_data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "No analysis generated.")
    )



# GET endpoint (text-only)

@app.get("/environmental_intelligence", response_model=AIResponse)
async def get_environmental_intelligence(
    lat: float = Query(..., ge=-90, le=90, description="Latitude of the location."),
    lon: float = Query(..., ge=-180, le=180, description="Longitude of the location."),
    start: date = Query("2025-01-01", description="Start date for analysis (YYYY-MM-DD)."),
    end: date = Query("2025-01-07", description="End date for analysis (YYYY-MM-DD)."),
    query: str = Query("summarize climate risks", description="Question or analysis request for prediction."),
):
    nasa_summary = await fetch_nasa_power_data(lat, lon, start, end)

    summary_text = (
        f"Location: lat={lat}, lon={lon}\n"
        f"Date range: {start} to {end}\n"
        f"Temperature (°C): {list(nasa_summary['T2M'].values())}\n"
        f"Precipitation (mm/day): {list(nasa_summary['PRECTOT'].values())}\n"
        f"Wind Speed (m/s): {list(nasa_summary['WS2M'].values())}\n"
        f"Relative Humidity (%): {list(nasa_summary['RH2M'].values())}\n"
        f"User Query: {query}"
    )

    ai_text = await call_gemini_api(summary_text)

    return {
        "request_latitude": lat,
        "request_longitude": lon,
        "nasa_data_used": nasa_summary,
        "user_query": query,
        "ai_environmental_analysis": ai_text,
    }



# POST endpoint (image upload)

@app.post("/environmental_intelligence", response_model=AIResponse)
async def post_environmental_intelligence(
    lat: float = Form(..., description="Latitude of the location."),
    lon: float = Form(..., description="Longitude of the location."),
    start: date = Form(..., description="Start date for analysis (YYYY-MM-DD)."),
    end: date = Form(..., description="End date for analysis (YYYY-MM-DD)."),
    query: str = Form("analyze environment with image", description="User query for prediction."),
    file: UploadFile = File(..., description="Optional image file to include in the analysis."),
):
    nasa_summary = await fetch_nasa_power_data(lat, lon, start, end)

    # Convert uploaded image to base64
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    summary_text = (
        f"Location: lat={lat}, lon={lon}\n"
        f"Date range: {start} to {end}\n"
        f"NASA Climate Data: {nasa_summary}\n"
        f"User Query: {query}\n"
        f"Analyze this together with the uploaded image."
    )

    ai_text = await call_gemini_api(summary_text, image_base64=image_base64)

    return {
        "request_latitude": lat,
        "request_longitude": lon,
        "nasa_data_used": nasa_summary,
        "user_query": query,
        "ai_environmental_analysis": ai_text,
    }
