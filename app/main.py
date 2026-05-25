""""
FastApi and gradio serving application - production ready ML model serving

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr


# Intialize the FastApI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting the churn in telecom industry",
    version="1.0.0",
)

# Health chek-up endpoint (for aws application load balancer health checks)
@app.get("/")
def root():
    """
      Health check endpoint for monitoring and load balancer health checks.
      """
    return {"status": "ok"}

# Request data schema
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
       Customer data schema for churn prediction.

       This schema defines the exact 18 features required for churn prediction.
       All features match the original dataset structure for consistency.
       """

    # Demographics
    gender: str   # Male or Female
    Partner: str   # Yes or no (same for below)
    Dependents: str

    # Phone services
    PhoneService: str   # Yes or No
    MultipleLines: str  # Yes or No or No phone service

    # Internet Services
    InternetService: str  # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str  # "Yes", "No", or "No internet service" (same for all below)
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    # Account Information
    Contract: str          # "Month-to-month", "One year", "Two year"
    PaperLessBilling: str    # Yes or No
    PaymentMethod: str        # "Electronic check", "Mailed check", etc.

    # Numeric features
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


##### Main  endpoint [predict]
@app.get('/predict')
def get_prediction():
    """
        Main prediction endpoint for customer churn prediction.

        This endpoint:
        1. Receives validated customer data via Pydantic model
        2. Calls the inference pipeline to transform features and predict
        3. Returns churn prediction in JSON format

        Expected Response:
        - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
        - {"error": "error_message"} if prediction fails
        """
    try:
        # Covert the pydantic model to dictionary
        result = predict()