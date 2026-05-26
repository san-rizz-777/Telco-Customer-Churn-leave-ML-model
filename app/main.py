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
import sys
import os

# Fix the import path
sys.path.append(os.path.abspath("src"))
from serving.inference import predict

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
def get_prediction(data: CustomerData):
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
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        # For debugging purpose return the error
        return {"error":str(e)}



    ###############################################

    # gradio web interface
def gradio_interface( gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges):

    """
     Gradio interface function that processes form inputs and returns prediction.

    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    """

    # Construct data dictionary matching CustomerData schema
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),  # Ensure integer type
        "MonthlyCharges": float(MonthlyCharges),  # Ensure float type
        "TotalCharges": float(TotalCharges),  # Ensure float type
    }

    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# gradio ui configuration
# Build comprehensive Gradio interface with all customer features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics section
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),

        # Phone services section
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),

        # Internet services section (key churn predictors)
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),

        # Contract and billing section (major churn factors)
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),

        # Numeric features (important for churn prediction)
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**

    Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on 
    historical telecom customer data to identify customers at risk of churning.

    **Tip**: Month-to-month contracts with fiber optic internet and electronic check payments 
    tend to have higher churn rates.
    """,
    examples=[
        # High churn risk example
        ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No",
         "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check",
         1, 85.0, 85.0],
        # Low churn risk example
        ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
         "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
         60, 45.0, 2700.0]
    ],
    theme=gr.themes.Soft()  # Professional appearance
)


# mount the gradio ui into fastapi
# creates the /ui endpoint that serves the Gradio interface
app = gr.mount_gradio_app(app, demo, url='/ui')         # fastapi instance , gradio interface and url accesible to it



