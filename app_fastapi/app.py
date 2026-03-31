import os
import uuid
import shutil
import csv
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from HINT.dataloader import csv_three_feature_2_dataloader
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.molecule_encode import MPNN, ADMET 
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import HINTModel 


device = torch.device("cpu")

print("==================================================")
print(" HINT FastAPI Server Initializing... ")
print("==================================================")

base_name = 'phase_II'
hint_model_path = os.path.join("save_model", f"{base_name}.ckpt")

# Load the AI model into memory (runs once at server startup)
if not os.path.exists(hint_model_path):
    raise Exception(f"Model not found at {hint_model_path}. Please train or download the model first.")

print(f"Loading '{base_name}' model from {hint_model_path}...")
model = torch.load(hint_model_path, map_location=device)
model.eval()
print("Model loading complete! Server is ready.")

app = FastAPI(title="HINT Clinical Trial Prediction API")


@app.get("/")
async def get_index():
    return FileResponse("index.html")

# # --- 메인 예측 엔드포인트 (기존 predict_manual을 대체) ---
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # 1. Save uploaded file temporarily
#     os.makedirs("data/temp", exist_ok=True)
#     temp_file_path = f"data/temp/{file.filename}"
    
#     with open(temp_file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     # 2. Convert CSV to dataloader
#     print(f"\n--- File {file.filename} received. Processing... ---")
#     test_loader = csv_three_feature_2_dataloader(temp_file_path, shuffle=False, batch_size=32)
    
#     # 3. Model Inference
#     # ongoing_test returns lists of nctid_all and predict_all
#     nctid_all, predict_all = model.ongoing_test(test_loader)
    
#     # 4. Format Output Json
#     results = []
#     for nctid, prob in zip(nctid_all, predict_all):
#         results.append({
#             "nctid": nctid,
#             "success_probability": round(float(prob), 4),
#             "prediction_label": 1 if float(prob) >= 0.5 else 0
#         })
        
#     # 5. Clean up temporary file
#     try:
#         os.remove(temp_file_path)
#     except:
#         pass
        
#     return {"status": "success", "total_records": len(results), "results": results}

class ManualTrialData(BaseModel):
    phase: str
    diseases: str
    icdcodes: str
    drugs: str
    smiless: str
    criteria: str

@app.post("/predict_manual")
async def predict_manual(data: ManualTrialData):
    # 1. Create Temporary File based on manual input
    os.makedirs("data/temp", exist_ok=True)
    # 내부 처리를 위한 고유 ID 생성 (파일명 중복 방지)
    internal_id = str(uuid.uuid4())
    temp_file_path = f"data/temp/{internal_id}.csv"
    # 1. HINT 모델 전처리기(dataloader)가 인식할 수 있는 최소한의 임시 CSV 생성
    with open(temp_file_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # 모델이 기대하는 10개의 컬럼 헤더 유지
        writer.writerow(["nctid","status","why_stop","label","phase","diseases","icdcodes","drugs","smiless","criteria"])
        # nctid 자리에는 내부 ID를 넣고, 나머지는 사용자 입력값 배치
        writer.writerow([internal_id, "completed", "", 0, data.phase, data.diseases, data.icdcodes, data.drugs, data.smiless, data.criteria])
            
    # 2. Convert CSV to dataloader
    test_loader = csv_three_feature_2_dataloader(temp_file_path, shuffle=False, batch_size=32)
    
    # 3. 모델 추론
    # nctid_all은 내부적으로만 사용하고 결과에서는 노출하지 않음
    _, predict_all = model.ongoing_test(test_loader)
    
    # 4. 결과 정리
    prob = float(predict_all[0]) # 단일 예측이므로 첫 번째 값 추출
    results = {
            "success_probability": round(prob, 4),
            "prediction_label": 1 if prob >= 0.5 else 0
    }
        
    # 4. Clean up
    try:
        os.remove(temp_file_path)
    except:
        pass
        
    return {"status": "success",  "results": results}

if __name__ == "__main__":
    import uvicorn
    print("\n[INFO] Starting web server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
