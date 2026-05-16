import json
import urllib.request
import urllib.error
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path so we can import src.predict
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_readmission

# ============================================================
# FastAPI App Configuration
# ============================================================
app = FastAPI(
    title="Hospital Readmission Risk Scorer",
    description=(
        "AI-Powered Clinical Decision Support System for predicting "
        "whether a diabetic patient will be readmitted within 30 days of discharge. "
        "Uses an optimized XGBoost model trained on 101,766 patient records."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================
class PatientInput(BaseModel):
    """Patient data input schema for readmission prediction."""

    # Required clinical fields
    time_in_hospital: int = Field(..., ge=1, le=30, description="Days in hospital (1-30)")
    num_lab_procedures: int = Field(..., ge=0, description="Number of lab procedures")
    num_procedures: int = Field(..., ge=0, description="Number of non-lab procedures")
    num_medications: int = Field(..., ge=0, description="Number of distinct medications")
    number_outpatient: int = Field(..., ge=0, description="Outpatient visits in past year")
    number_emergency: int = Field(..., ge=0, description="Emergency visits in past year")
    number_inpatient: int = Field(..., ge=0, description="Inpatient visits in past year")
    number_diagnoses: int = Field(..., ge=1, description="Number of diagnoses")

    # Demographics
    patient_name: Optional[str] = Field(None, description="Name of the patient")
    doctor_name: Optional[str] = Field(None, description="Name of the attending physician")
    age: str = Field(..., description="Age bracket, e.g. '[50-60)'",
                     pattern=r"^\[\d+-\d+\)$")
    race: Optional[str] = Field("Caucasian", description="Patient race")
    gender: Optional[str] = Field("Female", description="Patient gender")

    # Admission details
    admission_type_id: int = Field(..., ge=1, le=8, description="Admission type (1-8)")
    discharge_disposition_id: int = Field(..., ge=1, le=30, description="Discharge disposition")
    admission_source_id: int = Field(..., ge=1, le=26, description="Admission source")

    # Diagnosis codes (ICD-9)
    diag_1: str = Field(..., description="Primary diagnosis ICD-9 code")
    diag_2: str = Field(..., description="Secondary diagnosis ICD-9 code")
    diag_3: str = Field(..., description="Tertiary diagnosis ICD-9 code")

    # Diabetes care features
    A1Cresult: Optional[str] = Field("None", description="A1C result: None, Norm, >7, >8")
    max_glu_serum: Optional[str] = Field("None", description="Glucose serum: None, Norm, >200, >300")
    insulin: Optional[str] = Field("No", description="Insulin: No, Down, Steady, Up")
    change: Optional[str] = Field("No", description="Medication change: Ch or No")
    diabetesMed: Optional[str] = Field("No", description="Diabetes medication: Yes or No")

    # Medication columns (all optional, default to 'No')
    metformin: Optional[str] = Field("No")
    repaglinide: Optional[str] = Field("No")
    nateglinide: Optional[str] = Field("No")
    chlorpropamide: Optional[str] = Field("No")
    glimepiride: Optional[str] = Field("No")
    acetohexamide: Optional[str] = Field("No")
    glipizide: Optional[str] = Field("No")
    glyburide: Optional[str] = Field("No")
    tolbutamide: Optional[str] = Field("No")
    pioglitazone: Optional[str] = Field("No")
    rosiglitazone: Optional[str] = Field("No")
    acarbose: Optional[str] = Field("No")
    miglitol: Optional[str] = Field("No")
    troglitazone: Optional[str] = Field("No")
    tolazamide: Optional[str] = Field("No")
    examide: Optional[str] = Field("No")
    citoglipton: Optional[str] = Field("No")
    glyburide_metformin: Optional[str] = Field("No", alias="glyburide-metformin")
    glipizide_metformin: Optional[str] = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Optional[str] = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Optional[str] = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: Optional[str] = Field("No", alias="metformin-pioglitazone")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "time_in_hospital": 5,
                "num_lab_procedures": 44,
                "num_procedures": 1,
                "num_medications": 13,
                "number_outpatient": 0,
                "number_emergency": 0,
                "number_inpatient": 2,
                "number_diagnoses": 7,
                "age": "[50-60)",
                "race": "Caucasian",
                "gender": "Female",
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "diag_1": "250.83",
                "diag_2": "276",
                "diag_3": "250",
                "A1Cresult": ">8",
                "max_glu_serum": "Norm",
                "insulin": "Up",
                "change": "Ch",
                "diabetesMed": "Yes"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output schema."""
    risk_score: float = Field(..., description="Readmission probability (0.0 to 1.0)")
    risk_level: str = Field(..., description="Risk category: High, Medium, or Low")
    details: dict = Field(..., description="Threshold and model info")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    version: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: bool = True
    message: str
    errors: Optional[List[str]] = None


class AnalyzeRequest(BaseModel):
    """Request schema for OpenRouter AI analysis."""
    risk_score: float
    patient: dict


class AnalyzeResponse(BaseModel):
    """Response schema for OpenRouter AI analysis."""
    summary: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    patient: dict
    treatment_plan: str


class ChatResponse(BaseModel):
    reply: str


class UserAuth(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class AuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


# ============================================================
# Auth Helpers
# ============================================================
USERS_FILE = PROJECT_ROOT / "data" / "users.json"

def load_users():
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


# ============================================================
# Helper Functions
# ============================================================
# ============================================================
# AI Helper (OpenRouter API)
# ============================================================
def call_clinical_ai(prompt: str, history: List[Dict[str, str]] = None) -> str:
    """Helper to call OpenRouter API with error handling and fallbacks."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "The AI clinical assistant is unavailable (API key missing). Please review manual recommendations."
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    model = os.environ.get("AI_MODEL", "google/gemini-2.0-flash-001")
    
    messages = []
    if history:
        for msg in history:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
                'HTTP-Referer': 'https://hospital-readmission-scorer.ai',
                'X-Title': 'Hospital Readmission Scorer'
            }
        )
        
        with urllib.request.urlopen(req, timeout=25) as response:
            res_content = response.read().decode('utf-8')
            res_data = json.loads(res_content)
            if "choices" in res_data and res_data["choices"]:
                choice = res_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            
            return "Analysis failed: No content returned from AI model."
            
    except Exception as e:
        print(f"OpenRouter API Error: {str(e)}")
        return "The AI clinical assistant is temporarily unavailable. Please review the manual recommendations or try again in a few minutes."


# ============================================================
# API Endpoints
# ============================================================

@app.post("/register", response_model=AuthResponse, tags=["Auth"])
async def register(auth: UserAuth):
    """Register a new user."""
    users = load_users()
    if auth.email in users:
        return {"success": false, "message": "User already exists"}
    
    users[auth.email] = {
        "email": auth.email,
        "password": auth.password, # In a real app, hash this!
        "name": auth.name or auth.email.split("@")[0].capitalize()
    }
    save_users(users)
    
    user_data = users[auth.email].copy()
    del user_data["password"]
    
    return {"success": True, "message": "Registration successful", "user": user_data}


@app.post("/login", response_model=AuthResponse, tags=["Auth"])
async def login(auth: UserAuth):
    """Login an existing user."""
    users = load_users()
    user = users.get(auth.email)
    
    if not user or user["password"] != auth.password:
        return {"success": False, "message": "Invalid email or password"}
    
    user_data = user.copy()
    del user_data["password"]
    
    return {"success": True, "message": "Login successful", "user": user_data}


@app.get("/", tags=["Info"])
async def root():
    """API root — basic info and links."""
    return {
        "name": "Hospital Readmission Risk Scorer API",
        "version": "1.0.0",
        "description": "Predicts 30-day readmission risk for diabetic patients",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict (POST)",
            "analyze": "/analyze (POST)",
            "treatment_plan": "/treatment_plan (POST)",
            "chat": "/chat (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check if the API and model are loaded and ready."""
    return {
        "status": "healthy",
        "model": "XGBoost (optimized)",
        "version": "1.0.0"
    }


@app.post("/predict",
          response_model=PredictionResponse,
          responses={
              400: {"model": ErrorResponse, "description": "Invalid input"},
              500: {"model": ErrorResponse, "description": "Prediction error"}
          },
          tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Predict 30-day hospital readmission risk for a diabetic patient.
    """
    try:
        # Use model_dump for Pydantic v2, dict for v1
        if hasattr(patient, "model_dump"):
            patient_dict = patient.model_dump(by_alias=True)
        else:
            patient_dict = patient.dict(by_alias=True)
            
        result = predict_readmission(patient_dict)

        if result.get('error'):
            print(f"Prediction logic error: {result.get('message')}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": True,
                    "message": result.get('message', 'Invalid input'),
                    "errors": result.get('errors', [])
                }
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Prediction crash: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": True,
                "message": f"Prediction failed: {str(e)}",
                "errors": [str(e)]
            }
        )


@app.post("/predict/batch",
          response_model=List[PredictionResponse],
          tags=["Prediction"])
async def predict_batch(patients: List[PatientInput]):
    """
    Predict readmission risk for multiple patients at once.
    """
    if len(patients) > 50:
        raise HTTPException(
            status_code=400,
            detail={"error": True, "message": "Maximum 50 patients per batch request"}
        )

    results = []
    for i, p in enumerate(patients):
        try:
            p_dict = p.model_dump(by_alias=True)
            res = predict_readmission(p_dict)
            if res.get('error'):
                raise HTTPException(
                    status_code=400,
                    detail={"error": True, "message": f"Patient {i}: {res.get('message')}"}
                )
            results.append(res)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": True, "message": f"Patient {i} failed: {str(e)}"}
            )
    return results


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_risk(data: AnalyzeRequest):
    """
    Generate a clinical summary using OpenRouter AI.
    """
    try:
        pct = round(data.risk_score * 100)
        p = data.patient
        
        prompt = (
            f"As a clinical AI assistant, provide a comprehensive analysis for a diabetic patient with a readmission risk score of {pct}%. "
            f"Patient Profile: Age {p.get('age', 'N/A')}, Gender {p.get('gender', 'N/A')}, Race {p.get('race', 'N/A')}. "
            f"Clinical Indicators: A1C {p.get('A1Cresult', 'N/A')}, Inpatient visits {p.get('number_inpatient', 'N/A')}, "
            f"Emergency visits {p.get('number_emergency', 'N/A')}, Medications {p.get('num_medications', 'N/A')}, "
            f"Length of Stay {p.get('time_in_hospital', 'N/A')} days. "
            f"Instructions: Generate a professional 3-4 sentence clinical summary. "
            f"1. Identify the primary risk drivers. 2. Provide specific, actionable clinical recommendations for the attending physician. "
            f"3. Outline a preventative post-discharge action plan. "
            f"Maintain a formal medical tone. Use markdown bolding for key terms but keep the overall length concise."
        )
        
        summary = call_clinical_ai(prompt)
        return {"summary": summary}
    except Exception as e:
        print(f"Error in /analyze: {str(e)}")
        return {"summary": f"Analysis temporarily unavailable due to a technical error. Please use manual risk factor assessment."}


@app.post("/treatment_plan", response_model=AnalyzeResponse, tags=["Analysis"])
async def treatment_plan(data: AnalyzeRequest):
    """
    Generate a detailed clinical treatment plan using OpenRouter AI.
    """
    try:
        pct = round(data.risk_score * 100)
        p = data.patient
        
        prompt = (
            f"As a clinical AI assistant, analyze a diabetic patient with a hospital readmission risk score of {pct}%. "
            f"Patient details: Age {p.get('age', 'N/A')}, Gender {p.get('gender', 'N/A')}, A1C {p.get('A1Cresult', 'N/A')}, "
            f"Inpatient visits {p.get('number_inpatient', 'N/A')}, Medications {p.get('num_medications', 'N/A')}, "
            f"Time in hospital {p.get('time_in_hospital', 'N/A')} days. "
            f"Provide a comprehensive, step-by-step treatment plan and preventative action plan to avoid readmission. "
            f"Structure the response into clear sections: "
            f"1) Immediate Actions, "
            f"2) Medication Management, "
            f"3) Lifestyle & Small Diet Plan (provide specific diabetic-friendly meal suggestions), "
            f"4) Warning Signs & Red Flags, and "
            f"5) Follow-up Schedule. "
            f"Keep it professional, clinical, and directly address the risk factors. Please format it with Markdown to be easily readable."
        )
        
        plan = call_clinical_ai(prompt)
        return {"summary": plan}
    except Exception as e:
        print(f"Error in /treatment_plan: {str(e)}")
        return {"summary": "Treatment plan generation is currently unavailable. Please refer to standard clinical discharge protocols."}


@app.post("/chat", response_model=ChatResponse, tags=["Analysis"])
async def chat_with_plan(data: ChatRequest):
    """
    Chat with the AI regarding the generated treatment plan.
    """
    try:
        system_prompt = (
            f"You are a helpful clinical AI assistant discussing a patient's readmission risk and treatment plan. "
            f"Patient details: {data.patient}. "
            f"Suggested Treatment Plan: {data.treatment_plan}. "
            f"Answer the user's questions clearly, concisely, and professionally based on this context. "
            f"Do not give medical advice outside of this context."
        )
        
        # Initialize history with the system context
        full_history = [
            {"role": "user", "content": system_prompt},
            {"role": "model", "content": "Understood. I have reviewed the patient's data and the suggested treatment plan. I am ready to assist with clinical questions or clarifications."}
        ]
        
        # Append the conversation history from the user
        for msg in data.history:
            role = "user" if msg.role == "user" else "model"
            full_history.append({"role": role, "content": msg.content})
            
        reply = call_clinical_ai(data.message, history=full_history)
        return {"reply": reply}
    except Exception as e:
        print(f"Error in /chat: {str(e)}")
        return {"reply": "I'm sorry, I'm having trouble connecting to the clinical knowledge base right now. Please try again in a moment."}


@app.get("/model/info", tags=["Info"])
async def model_info():
    """Get information about the trained model."""
    return {
        "model_type": "XGBoost Classifier",
        "optimization": "RandomizedSearchCV with F2-score",
        "dataset": "Diabetic Patient Readmission Dataset",
        "dataset_size": 101766,
        "features": 124,
        "metrics": {
            "recall": 0.8230,
            "f1_score": 0.2519,
            "f2_score": 0.4316,
            "roc_auc": 0.6871,
            "accuracy": 0.4547
        },
        "risk_thresholds": {
            "high": ">= 0.7",
            "medium": "0.4 - 0.69",
            "low": "< 0.4"
        },
        "primary_metric": "Recall (minimizing missed high-risk patients)"
    }
