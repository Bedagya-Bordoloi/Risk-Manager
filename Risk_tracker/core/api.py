from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os

from main import run_analysis_service
from core.manual_entry import add_transaction, add_invoice

app = FastAPI()

UPLOAD_PATH = "uploads"


# ---------------------------------------------------
# ROOT
# ---------------------------------------------------

@app.get("/")
def home():
    return {"message": "AI Risk Manager API Running"}


# ---------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------

@app.get("/analysis")
def get_analysis():

    result = run_analysis_service()

    return result


# ---------------------------------------------------
# UPLOAD DATASET
# ---------------------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_location = os.path.join(UPLOAD_PATH, "user_data.csv")

    df = pd.read_csv(file.file)

    df.to_csv(file_location, index=False)

    result = run_analysis_service()

    return result


# ---------------------------------------------------
# MANUAL TRANSACTION ENTRY
# ---------------------------------------------------

@app.post("/add_transaction")
def api_add_transaction(data: dict):

    add_transaction(
        data["client_id"],
        data["amount"],
        data["date"],
        data.get("type", "income"),
        data.get("category", "Service")
    )

    return {"status": "transaction added"}


# ---------------------------------------------------
# MANUAL INVOICE ENTRY
# ---------------------------------------------------

@app.post("/add_invoice")
def api_add_invoice(data: dict):

    add_invoice(
        data["client_id"],
        data["due_date"],
        data["paid_date"],
        data["amount"]
    )

    return {"status": "invoice added"}
