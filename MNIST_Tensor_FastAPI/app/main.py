from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException

import numpy as np

app = FastAPI(title="MNIST Tensor API", description="A simple API for MNIST Tensor")

class DigitInput(BaseModel):
    pixels:List[float]

@app.post("/predict")
def predict(data: DigitInput):
    image_array = np.array(data.pixels, dtype=np.float32)

    if image_array.shape[0] != 784:
        return {"error": "Input must be a list of 784 float values."}
    
    try:
        image_np = image_array.reshape((1, 28, 28, 1))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reshaping input: {str(e)}")


    prediction = 5

    return {"Predicted Digit: ", prediction}