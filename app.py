from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from typing import Dict, Tuple, Any
import logging
from pathlib import Path
import os

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
            
            # Attempt to load CSV with better error handling
            try:
                self.df = pd.read_csv(file_path)
            except pd.errors.ParserError as e:
                logger.error(f"Error loading CSV file: {str(e)}")
                raise ValueError(f"Invalid CSV format in file: {file_path}. Please check the file.")
            
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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Root endpoint with a simple HTML form for drug interaction prediction"""
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drug Interaction Prediction</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f7fc;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }

        .container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            padding: 30px;
            width: 100%;
            max-width: 600px;
        }

        h1 {
            text-align: center;
            color: #4CAF50;
            font-size: 28px;
            margin-bottom: 20px;
        }

        label {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 8px;
            display: block;
        }

        input {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            box-sizing: border-box;
        }

        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 4px;
            width: 100%;
            margin-top: 10px;
            transition: background-color 0.3s ease;
        }

        button:hover {
            background-color: #45a049;
        }

        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .result strong {
            color: #333;
        }

        .error {
            color: red;
            font-weight: bold;
        }

        .success {
            color: #FC1313FF;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>Drug Interaction Prediction</h1>
        <form id="drugForm">
            <label for="drug1">Drug 1:</label>
            <input type="text" id="drug1" name="drug1" placeholder="Enter first drug name" required><br>

            <label for="drug2">Drug 2:</label>
            <input type="text" id="drug2" name="drug2" placeholder="Enter second drug name" required><br>

            <button type="submit">Predict Interaction</button>
        </form>

        <div class="result" id="result"></div>
    </div>

    <script>
        document.getElementById('drugForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const drug1 = document.getElementById('drug1').value;
            const drug2 = document.getElementById('drug2').value;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ drug1: drug1, drug2: drug2 })
            });

            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = ''; // Clear previous result

            if (response.ok) {
                const data = await response.json();
                resultDiv.innerHTML = `
                    <div class="success">
                        <strong>Drug Interaction:</strong> ${data.description}<br>
                        <strong>Severity:</strong> ${data.severity}<br>
                        <strong>Confidence Score:</strong> ${data.confidence_score}
                    </div>
                `;
            } else {
                const errorData = await response.json();
                resultDiv.innerHTML = `<div class="error">${errorData.detail}</div>`;
            }
        });
    </script>

</body>
</html>
    """

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not model_manager.model_description or not model_manager.model_severity:
        raise HTTPException(status_code=503, detail="Models not initialized")
    return {"status": "healthy", "models_loaded": True}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
