from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from logger import get_logger

# Create FastAPI app
logger = get_logger(__name__)
app = FastAPI(title="Placement Prediction API")

# Load model & scaler at startup
model = joblib.load("models/xgboost_pipeline.pkl")
#scaler = joblib.load("model/scaler.pkl")

# Define input schema
class PredictionInput(BaseModel):
    student_id	:str
    gender	:str
    age	:int
    city_tier	:str
    ssc_percentage	:float
    ssc_board	:str
    hsc_percentage	:float
    hsc_board	:str
    hsc_stream	:str
    degree_percentage	:float
    degree_field	:str
    mba_percentage	:float
    specialization	:str
    internships_count	:int
    projects_count	:int
    certifications_count	:int
    technical_skills_score	:float
    soft_skills_score	:float
    aptitude_score	:float
    communication_score	:float
    work_experience_months	:int
    leadership_roles	:int
    extracurricular_activities	:int
    backlogs	:int
    salary_lpa	:float
# Home route
@app.get("/")
def home():
    return {"message": "FastAPI ML model is running"}

@app.get("/")
def root():
    return {"message": "Send POST requests to /predict for predictions"}

# Prediction route
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convert input to numpy array
        logger.info("Received prediction request")
        input_data = pd.DataFrame([{"student_id"	:	data.student_id,
"gender"	:	data.gender,
"age"	:	data.age,
"city_tier"	:	data.city_tier,
"ssc_percentage"	:	data.ssc_percentage,
"ssc_board"	:	data.ssc_board,
"hsc_percentage"	:	data.hsc_percentage,
"hsc_board"	:	data.hsc_board,
"hsc_stream"	:	data.hsc_stream,
"degree_percentage"	:	data.degree_percentage,
"degree_field"	:	data.degree_field,
"mba_percentage"	:	data.mba_percentage,
"specialization"	:	data.specialization,
"internships_count"	:	data.internships_count,
"projects_count"	:	data.projects_count,
"certifications_count"	:	data.certifications_count,
"technical_skills_score"	:	data.technical_skills_score,
"soft_skills_score"	:	data.soft_skills_score,
"aptitude_score"	:	data.aptitude_score,
"communication_score"	:	data.communication_score,
"work_experience_months"	:	data.work_experience_months,
"leadership_roles"	:	data.leadership_roles,
"extracurricular_activities"	:	data.extracurricular_activities,
"backlogs"	:	data.backlogs,
"salary_lpa"	:	data.salary_lpa}])


        # Apply scaling
        #scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)
                                  
        logger.info("Prediction Done Successfully")
        return {"prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"Error occured: {e}")
        raise HTTPException(status_code=500, detail=str(e))
