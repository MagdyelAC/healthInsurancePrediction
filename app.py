from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib

from fastapi import FastAPI

app = FastAPI(title = 'Health Insurance Cost Prediction')

model = load(pathlib.Path('model/insurance-v1.joblib'))

class InputData(BaseModel):
    age :int= 40
    sex :int= 1
    bmi :int= 40.30
    children :int= 4
    smoker :int= 1
    region :int= 2

class OutputData(BaseModel):
    prediction:float=42488.828178

@app.post('/prediction', response_model = OutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict(model_input)

    return {'prediction':result}
