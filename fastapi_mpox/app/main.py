from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import zipfile
import io
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix, f1_score
from .inference import predict
from .schemas import PredictionResponse

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    result = predict(content)
    return PredictionResponse(**result)

@app.post("/predict/bulk")
async def predict_bulk(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        result = predict(content)
        results.append({
            "filename": file.filename,
            "prediction": result
        })
    return {"results": results}

@app.post("/predict/evaluate")
async def evaluate_zip(file: UploadFile = File(...)):
    print("=" * 50)
    print("Starting ZIP evaluation...")
    print("=" * 50)
    
    content = await file.read()
    print(f"ZIP file size: {len(content) / 1024 / 1024:.2f} MB")
    
    # Define expected class names (must match training)
    VALID_CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
    
    results = []
    true_labels = []
    pred_labels = []
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
        
        # Debug: show folder structure
        sample_files = all_files[:10]
        print("Sample file paths (first 10):")
        for f in sample_files:
            print(f"  - {f}")
        
        for i, name in enumerate(all_files):
            try:
                # Get true label from folder name
                parts = name.split('/')
                
                # Find class folder - look for any part that matches our valid classes
                true_label = None
                for part in parts:
                    # Normalize: lowercase, strip
                    part_normalized = part.strip()
                    # Check if this part matches any valid class (case-insensitive)
                    for valid_cls in VALID_CLASSES:
                        if part_normalized.lower() == valid_cls.lower():
                            true_label = valid_cls  # Use exact case from VALID_CLASSES
                            break
                    if true_label:
                        break
                
                if true_label is None:
                    print(f"Skipping (no valid label): {name}")
                    skipped += 1
                    continue
                
                print(f"Processing: {name} | True label: {true_label}")
                
                # Read and predict
                img_content = zf.read(name)
                pred = predict(img_content)
                predicted_label = pred['label']

                print("Prediction....", predicted_label)
                
                total += 1
                per_class[true_label]["total"] += 1
                true_labels.append(true_label)
                pred_labels.append(predicted_label)
                
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
                
                # Print progress every 10 images
                if total % 10 == 0:
                    current_acc = (correct / total * 100)
                    print(f"Progress: {total}/{len(all_files)} | Current Accuracy: {current_acc:.2f}%")
                    
            except Exception as e:
                print(f"Error processing {name}: {e}")
                skipped += 1
                continue
    
    if skipped > 0:
        print(f"Skipped {skipped} files")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    class_accuracy = {
        cls: (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
        for cls, data in per_class.items()
    }
    
    cm = confusion_matrix(true_labels, pred_labels, labels=VALID_CLASSES)
    f1_macro = f1_score(true_labels, pred_labels, labels=VALID_CLASSES, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, labels=VALID_CLASSES, average='weighted', zero_division=0)
    f1_per_class = f1_score(true_labels, pred_labels, labels=VALID_CLASSES, average=None, zero_division=0)
    f1_scores = {cls: round(score, 4) for cls, score in zip(VALID_CLASSES, f1_per_class)}
    
    print("=" * 50)
    print(f"Evaluation Complete!")
    print(f"Total Images: {total}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print("Per-class Accuracy:")
    for cls, acc in class_accuracy.items():
        print(f"  - {cls}: {per_class[cls]['correct']}/{per_class[cls]['total']} = {acc:.2f}%")
    print("=" * 50)
    
    return {
        "total_images": total,
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "per_class_accuracy": class_accuracy,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": VALID_CLASSES,
        "f1_score_macro": round(f1_macro, 4),
        "f1_score_weighted": round(f1_weighted, 4),
        "f1_score_per_class": f1_scores,
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
