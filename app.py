from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware


from fastapi import FastAPI

origins = ['*']


app = FastAPI(title = 'Health Insurance Cost Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=['*'],
   allow_headers=['*']
)

model = load(pathlib.Path('model/insurance-v1.joblib'))

class InputData(BaseModel):
    age :int= 40
    sex :int= 1
    bmi : float= 40.30
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
