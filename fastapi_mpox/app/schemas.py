from pydantic import BaseModel
from typing import Dict

class UploadImage(BaseModel):
    # FastAPI will receive the file via FormData, so we don't need fields here.
    # This model exists for documentation purposes.
    pass

class PredictionResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]
