from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import zipfile
import io
import os
from collections import defaultdict
from .inference import predict, MODELS_DIR
from .schemas import PredictionResponse

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/models")
async def list_models():
    if not os.path.exists(MODELS_DIR):
        return {"models": []}
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(('.pt', '.pth'))]
    return {"models": sorted(models)}

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...), model_name: str = Form(None)):
    content = await file.read()
    result = predict(content, model_name=model_name)
    return PredictionResponse(**result)

@app.post("/predict/bulk")
async def predict_bulk(files: list[UploadFile] = File(...), model_name: str = Form(None)):
    results = []
    for file in files:
        content = await file.read()
        result = predict(content, model_name=model_name)
        results.append({
            "filename": file.filename,
            "prediction": result
        })
    return {"results": results}

@app.post("/predict/evaluate")
async def evaluate_zip(file: UploadFile = File(...), model_name: str = Form(None)):
    print("=" * 50)
    print(f"Starting ZIP evaluation with model: {model_name or 'Default'}...")
    print("=" * 50)
    
    content = await file.read()
    print(f"ZIP file size: {len(content) / 1024 / 1024:.2f} MB")
    
    # Define expected class names (must match training)
    VALID_CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
    
    results = []
    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    skipped = 0
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif')
    
    with zipfile.ZipFile(io.BytesIO(content), 'r') as zf:
        all_files = [n for n in zf.namelist() 
                     if not n.endswith('/') 
                     and n.lower().endswith(valid_exts)
                     and '__macosx' not in n.lower()
                     and 'thumbs.db' not in n.lower()
                     and '.ds_store' not in n.lower()]
        
        print(f"Found {len(all_files)} images to process")
        
        for i, name in enumerate(all_files):
            try:
                # Get true label from folder name
                parts = name.split('/')
                
                # Find class folder - look for any part that matches our valid classes
                true_label = None
                for part in parts:
                    part_normalized = part.strip()
                    for valid_cls in VALID_CLASSES:
                        if part_normalized.lower() == valid_cls.lower():
                            true_label = valid_cls
                            break
                    if true_label:
                        break
                
                if true_label is None:
                    skipped += 1
                    continue
                
                # Read and predict
                img_content = zf.read(name)
                pred = predict(img_content, model_name=model_name)
                predicted_label = pred['label']
                
                total += 1
                per_class[true_label]["total"] += 1
                
                is_correct = true_label.lower() == predicted_label.lower()
                if is_correct:
                    correct += 1
                    per_class[true_label]["correct"] += 1
                
                results.append({
                    "filename": os.path.basename(name),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": is_correct,
                    "probabilities": pred['probabilities']
                })
                
                if total % 10 == 0:
                    current_acc = (correct / total * 100)
                    print(f"Progress: {total}/{len(all_files)} | Current Accuracy: {current_acc:.2f}%")
                    
            except Exception as e:
                print(f"Error processing {name}: {e}")
                skipped += 1
                continue
    
    accuracy = (correct / total * 100) if total > 0 else 0
    class_accuracy = {
        cls: (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        for cls, data in per_class.items()
    }
    
    return {
        "total_images": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "per_class_accuracy": class_accuracy,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
