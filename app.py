from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Dict, Tuple, Any
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Drug Interaction Prediction API",
              description="API for predicting drug interactions and their severity",
              version="1.0.0")

class ModelManager:
    def __init__(self):
        self.model_description: RandomForestClassifier = None
        self.model_severity: RandomForestClassifier = None
        self.le_drug1: LabelEncoder = None
        self.le_drug2: LabelEncoder = None
        self.le_description: LabelEncoder = None
        self.le_severity: LabelEncoder = None
        self.df: pd.DataFrame = None
        
    def load_and_train_model(self, file_path: str) -> None:
        """
        Load dataset and train the models with error handling
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            # Load the dataset
            self.df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['drug1_name', 'drug2_name', 'interaction_type', 'Severity']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Normalize and preprocess data
            self.df['drug1_name'] = self.df['drug1_name'].str.strip().str.lower()
            self.df['drug2_name'] = self.df['drug2_name'].str.strip().str.lower()
            
            # Initialize and fit label encoders
            self.le_drug1 = LabelEncoder()
            self.le_drug2 = LabelEncoder()
            self.le_severity = LabelEncoder()
            self.le_description = LabelEncoder()
            
            self.df['drug1_encoded'] = self.le_drug1.fit_transform(self.df['drug1_name'])
            self.df['drug2_encoded'] = self.le_drug2.fit_transform(self.df['drug2_name'])
            self.df['severity_encoded'] = self.le_severity.fit_transform(self.df['Severity'])
            self.df['description_encoded'] = self.le_description.fit_transform(self.df['interaction_type'])
            
            # Prepare features and targets
            X = self.df[['drug1_encoded', 'drug2_encoded']]
            y_description = self.df['description_encoded']
            y_severity = self.df['severity_encoded']
            
            # Train models
            self.model_description = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model_severity = RandomForestClassifier(n_estimators=100, random_state=42)
            
            self.model_description.fit(X, y_description)
            self.model_severity.fit(X, y_severity)
            
            logger.info("Models successfully loaded and trained")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

class DrugInteractionRequest(BaseModel):
    drug1: str
    drug2: str
    
    @validator('drug1', 'drug2')
    def validate_drug_names(cls, v):
        if not v.strip():
            raise ValueError("Drug name cannot be empty")
        return v.strip().lower()  # Normalize drug names

class DrugInteractionResponse(BaseModel):
    drug1: str
    drug2: str
    description: str
    severity: str
    confidence_score: float

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        model_manager.load_and_train_model('DDI_data.csv')
        logger.info("API startup complete - models loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize models: {str(e)}")
        raise

@app.post("/predict", response_model=DrugInteractionResponse)
async def predict_interaction(request: DrugInteractionRequest):
    """
    Predict drug interaction and severity
    """
    try:
        # Validate drug presence in dataset
        if request.drug1 not in model_manager.le_drug1.classes_ or \
           request.drug2 not in model_manager.le_drug2.classes_:
            unknown_drugs = []
            if request.drug1 not in model_manager.le_drug1.classes_:
                unknown_drugs.append(request.drug1)
            if request.drug2 not in model_manager.le_drug2.classes_:
                unknown_drugs.append(request.drug2)
            raise HTTPException(
                status_code=400,
                detail=f"Unknown drug(s): {', '.join(unknown_drugs)}. Please check the drug names."
            )

        # Encode drugs
        drug1_encoded = model_manager.le_drug1.transform([request.drug1])[0]
        drug2_encoded = model_manager.le_drug2.transform([request.drug2])[0]
        
        # Make predictions
        interaction_encoded = model_manager.model_description.predict([[drug1_encoded, drug2_encoded]])[0]
        severity_encoded = model_manager.model_severity.predict([[drug1_encoded, drug2_encoded]])[0]
        
        # Get prediction probabilities for confidence score
        confidence_score = max(model_manager.model_description.predict_proba([[drug1_encoded, drug2_encoded]])[0])
        
        # Decode predictions
        description = model_manager.le_description.inverse_transform([interaction_encoded])[0]
        severity = model_manager.le_severity.inverse_transform([severity_encoded])[0]
        
        return DrugInteractionResponse(
            drug1=request.drug1,
            drug2=request.drug2,
            description=description,
            severity=severity,
            confidence_score=round(float(confidence_score), 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.get("/")
async def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Drug Interaction Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict drug interactions and severity",
            "GET /health": "Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not model_manager.model_description or not model_manager.model_severity:
        raise HTTPException(status_code=503, detail="Models not initialized")
    return {"status": "healthy", "models_loaded": True}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
