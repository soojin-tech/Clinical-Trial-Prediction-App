# 🎨 Clinical Trial Prediction UI (Streamlit App)

본 폴더는 사용자가 임상시험 데이터를 입력하고, 두 가지 인공지능 모델(HINT & ML)의 예측 결과를 시각적으로 확인할 수 있는 **프론트엔드 서비스**를 포함하고 있습니다.

## 🌟 주요 기능
1. **ML Prediction**: 설계 특성 데이터를 바탕으로 한 통계적 성공 확률 예측
2. **SHAP Interpretation**: 모델이 왜 그런 결과를 내놓았는지 변수별 기여도(SHAP Value) 시각화
3. **Data Visualization**: 입력 데이터의 분포 및 타겟 변수와의 상관관계 차트 제공
4. **Manual Input Interface**: 사용자가 직접 임상 설계 요소를 입력하여 실시간 시뮬레이션 가능

## 📂 폴더 구조
- `merge_app.py`: HINT와 ML 모델을 통합하여 보여주는 메인 스트림릿 소스 코드
- `models/`: 로컬에서 실행되는 ML 모델 파일들 (`.pkl`, `.joblib`)
- `Preprocessing/`: 데이터 전처리를 위한 커스텀 모듈 및 스케일러
- `results/`: 분석 결과 및 로그 저장 폴더
- `requirements_ml.txt`: 본 서비스를 실행하기 위해 필요한 라이브러리 목록

## 🛠️ 실행 방법

> **주의**: 반드시 최상위 폴더에 있는 `app_fastapi` 서버가 먼저 실행 중이어야 HINT 예측 기능이 작동합니다.

1. **가상환경 활성화**
   ```bash
   conda activate ct_study