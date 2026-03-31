# HINT: 임상시험 성공 예측 AI 모델 및 FastAPI 웹 백엔드

본 레포지토리는 임상시험 프로토콜 텍스트, 타겟 질병(ICD), 약물 분자 구조(SMILES) 데이터들을 융합 구조로 분석하여, 해당 임상시험의 성공 여부를 예측하는 **HINT (Hierarchical Interaction Network)** 딥러닝 모델의 구동 환경입니다.

---

## 🚀 1. 빠른 시작 및 설치 가이드 (환경 설정)

해당 AI 모델 및 웹 서버를 정상적으로 구동하기 위한 환경 설정 방법입니다. 모델은 `Python 3.7.1` 기반으로 설계되었습니다.

1. **가상환경 시작 (Conda)**
   ```bash
   conda activate predict_drug_clinical_trial
   ```
   > *(만약 가상환경이 없다면 `conda create -n predict_drug_clinical_trial python=3.7.1` 로 생성)*

2. **의존성(패키지) 한번에 설치하기**
   동봉된 `requirements.txt` 파일을 이용해서 HINT 논문 모델과 FastAPI 웹 서버 구동에 필요한 모든 라이브러리를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 2. HINT 원본 코드 및 사전 학습 모델 설명

### `run_hint.py` 란?
개발자들이 제공한 터미널용 배치 실행(Evaluation) 스크립트입니다. `base_name` 변수에 모델 이름(`toy`, `phase_I` 등)을 적어놓고 실행하면, 대응하는 테스트 데이터(예: `data/toy_test.csv`)를 쭉 읽어들여서 정답률(ROC-AUC 등)과 개별 임상의 예측 확률을 콘솔에 텍스트로 찍어냅니다.

### `save_model/` 폴더 내의 파일들 (`.ckpt`)
이 파일들은 사용자의 PC에서 처음부터 며칠 밤을 새며 0부터 학습된 파일이 아닙니다! 논문 저자들이 거대한 서버 전용 GPU에서 전 세계 수만 건의 데이터(Phase I, II, III 등)를 **미리 학습시켜 놓은 똑똑한 뇌의 덤프 파일(Pre-trained Weights)** 들입니다.
* `phase_I.ckpt`: 임상 1상 결과를 잘 맞추도록 특화 훈련된 AI 파일
* `phase_II.ckpt`: 임상 2상 결과 예측에 특화 훈련된 AI 파일
* `phase_III.ckpt`: 임상 3상 결과 예측에 특화 훈련된 AI 파일
* `toy.ckpt`: 작은 데이터 샘플(Toy Data)을 위해 별도로 훈련된 간소화 모델

> 💡 **사용 팁**: `app.py`나 `run_hint.py` 코드 내부의 `base_name = 'toy'` 라고 적힌 부분을, 분석하고자 하는 임상 단계 성격에 맞춰 `'phase_III'` 등으로 바꾸면 실시간으로 분석가의 뇌를 교체해 적용할 수 있습니다.

---

## 🌐 3. FastAPI 웹 서버 구동하기 (`app.py`)

기존 터미널에서만 돌던 AI를 외부 통신 및 대화 방식의 모던 웹 어플리케이션으로 호출하기 위해 `app.py`가 구축되었습니다. CSV 파일을 웹 폼으로 업로드하면, AI가 분석 후 웹페이지 상에 즉시 JSON 형식으로 확률을 쏴줍니다.

### 서버 켜기
가상환경이 켜진 터미널에서 아래 명령어를 실행합니다.
```bash
uvicorn app:app --reload 
```
(`--reload` 옵션 때문에 코드를 수정하고 저장하면 알아서 서버가 재시작됩니다.)

### API 테스트 해보기 (개발자 UI)
서버가 켜져 있는 상태에서, 인터넷 브라우저에 접속합니다.
👉 **접속 주소**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

1. 화면에 나오는 `POST /predict` 녹색 탭을 클릭하여 엽니다.
2. 우측 상단의 **`Try it out`** 버튼을 누릅니다.
3. **`Choose File`** 버튼이 나타나면, 분석할 임상시험 엑셀(`csv`) 파일을 올립니다. (예: `data/toy_test.csv`)
4. 파란색 **`Execute`** 버튼을 누르면, 백그라운드 서버가 AI 뇌를 돌려 예측합니다.
5. 곧바로 하단 `Responses` 검은 창에 NCT 번호와 해당 임상의 성공 확률(%)이 나열된 예측값 리스트가 출력됩니다!
