# 🏥 임상시험 성공 예측 통합 플랫폼 (Clinical Trial Predictor)

본 프로젝트는 **HINT(Deep Learning)** 모델과 **ML(Logistic Regression/XGBoost)** 모델을 결합하여 임상시험의 성공 가능성을 다각도로 분석하는 통합 서비스입니다.

## 📂 프로젝트 구조
- `app_fastapi/`: HINT 모델 기반의 예측 API 서버 (FastAPI)
- `app_streamlit/`: 사용자 인터페이스 및 ML 기반 분석/SHAP 도출 (Streamlit)

## 🛠️ 설치 및 실행 방법

### 1. 환경 준비
본 프로젝트는 두 개의 가상환경 사용을 권장합니다.
- **HINT 전용:** `hint_env`
- **ML 전용:** `ct_study`

### 2. 백엔드 실행 (FastAPI)
```bash
cd app_fastapi
conda activate hint_env
uvicorn app:app --reload --port 8000