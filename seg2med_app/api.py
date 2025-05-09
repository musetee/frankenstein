from fastapi import FastAPI
from pydantic import BaseModel
import tutorial8_app

app = FastAPI()

class SimulationRequest(BaseModel):
    modality: str
    params: dict

@app.post("/simulate/")
def simulate(request: SimulationRequest):
    result = tutorial8_app.run_simulation(
        modality=request.modality,
        params=request.params
    )
    return {"result": result.tolist()}  # or return a file URL
