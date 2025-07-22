from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    data: list

@app.post("/infer")
def run_inference(input_data: InputData):
    return {"prediction": "inference simulated!"}

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}
