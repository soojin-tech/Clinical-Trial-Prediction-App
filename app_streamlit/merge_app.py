import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import shap
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="임상시험 성공 예측 통합 서비스", layout="wide")

# --- 사이드바: 모델 선택 ---
st.sidebar.title("🛠️ 모델 선택")
app_mode = st.sidebar.selectbox("사용할 모델을 선택하세요", 
    ["서비스 안내", "HINT 모델 (Deep Learning)", "ML 모델 (Logistic/XGBoost)"])

# --- 1. 서비스 안내 화면 ---
if app_mode == "서비스 안내":
    st.title("🏥 임상시험 성공 예측 통합 플랫폼")
    st.write("사용 목적에 맞는 모델을 선택하여 임상 성공 가능성을 예측해보세요.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" HINT 모델")
        st.info("SMILES, 질환명, Inclusion/Exclusion 등 비정형 데이터를 딥러닝으로 분석합니다.")
    with col2:
        st.subheader(" ML 모델 (SHAP)")
        st.info("정형화된 변수를 바탕으로 예측하며, SHAP을 통해 결과의 원인을 분석합니다.")

# --- 2. HINT 모델 입력 화면 ---
elif app_mode == "HINT 모델 (Deep Learning)":
    st.title(" HINT 모델 예측")
    
    # 입력 폼 (기존 FastAPI용 항목들)
    with st.form("hint_form"):
        phase = st.selectbox("Phase", ["phase 1", "phase 2", "phase 3", "phase4"])
        diseases = st.text_input("Diseases")
        icdcodes = st.text_input("ICD-10 Codes")
        drugs = st.text_input("Drugs")
        smiless = st.text_input("SMILES")
        criteria = st.text_area("Protocol_Criteria")
        submit = st.form_submit_button("HINT 분석 시작")
        
        if submit:
            # FastAPI 서버로 데이터 전송
            payload = {
                "nctid": f"ST-{np.random.randint(10000)}",
                "phase": phase,
                "diseases": f"['{diseases}']",
                "icdcodes": f"['{icdcodes}']",
                "drugs": f"['{drugs}']",
                "smiless" : f"['{smiless}']",
                "criteria": criteria,
            }

            try:
                response = requests.post("http://127.0.0.1:8000/predict_manual", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    if "results" in result:
                        prob = result["results"].get("success_probability", 0.0)
                        label = result["results"].get("prediction_label", 0)
                        
                        st.success(f" 예측 성공 확률: {prob*100:.1f}%")
                        st.progress(prob)
                        
                        # 라벨에 따른 추가 설명 (선택 사항)
                        if label == 1:
                            st.info("💡 예측 결과: **Expected Success** (긍정적)")
                        else:
                            st.warning("⚠️ 예측 결과: **High Risk** (주의 필요)")
                    else:
                        st.error("서버 응답 형식에 'results' 키가 없습니다.")
                else:
                    st.error(f"서버 응답 에러: {response.status_code}")                
            except Exception as e:
                st.error(f" 연결 오류: {e}")

# --- 3. ML 모델 입력 화면 ---
elif app_mode == "ML 모델 (Logistic/XGBoost)":
    st.title("📊 머신러닝 기반 분석 및 SHAP 도출")
    
    # 기존 Streamlit ct_study 코드를 여기에 복사/연동
    # === Page Configuration ===
    # st.set_page_config(page_title = "Clinical Trial Outcome Predictor", layout = "centered")
    st.title("Clinical Trial Outcome Predictor")

    # st.markdown("""
    # 이 도구는 **AACT(ClinicalTrials.gov)** 데이터베이스를 기반으로 학습된 머신러닝 모델을 사용하여 설계 특성을 바탕으로 **임상시험 완료 가능성(성공 또는 실패)**을 예측합니다.

    # 예측은 과거 임상시험 데이터의 통계적 패턴에서 도출됩니다.
    # """)

    # === Load Models & Thresholds (robust relative paths) ===
    REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root
    MODELS_DIR = REPO_ROOT / "models"
    RESULTS_DIR = REPO_ROOT / "results"

    @st.cache_resource
    def load_models():
        try:
            MODELS_DIR = "models"
            RESULTS_DIR = "results"

            logreg_model = joblib.load(f"{MODELS_DIR}/logreg_pipeline.pkl")
            # explainer_lr = joblib.load(f"{MODELS_DIR}/shap_explainer_logreg.pkl")
            with open(f"{MODELS_DIR}/X_cols_logreg.pkl", "rb") as f:
                X_cols_logreg = pickle.load(f)
            explainer_lr = shap.LinearExplainer(logreg_model.named_steps['clf'], np.zeros((1, len(X_cols_logreg))))

            xgb_model = joblib.load(f"{MODELS_DIR}/xgb_pipeline.pkl")
            explainer_xgb = joblib.load(f"{MODELS_DIR}/shap_explainer_xgb.pkl")
            with open(f"{MODELS_DIR}/X_cols_xgb.pkl", "rb") as f:
                X_cols_xgb = pickle.load(f)

            logreg_metrics = pd.read_csv(f"{RESULTS_DIR}/model_logreg/model_logreg_metrics.csv")
            xgb_metrics    = pd.read_csv(f"{RESULTS_DIR}/model_xgb/model_xgb_metrics.csv")
            best_thr_lr  = float(logreg_metrics["Best_threshold"].iloc[0])
            best_thr_xgb = float(xgb_metrics["Best_threshold"].iloc[0])
            

            return logreg_model, X_cols_logreg, xgb_model, X_cols_xgb, best_thr_lr, best_thr_xgb, explainer_lr, explainer_xgb

        except Exception as e:
            import traceback
            st.error("Model/metrics load failed:")
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            raise

    logreg_model, X_cols_logreg, xgb_model, X_cols_xgb, best_thr_lr, best_thr_xgb, explainer_lr, explainer_xgb = load_models()

    # === Mapping Dictionaries ===
    phase_map = {'Not Applicable': 'phase_not applicable', 'Phase 1': 'phase_1', 'Phase 2': 'phase_2', 'Phase 3': 'phase_3', 'Phase 4': 'phase_4'}

    intervention_map = {
        'Behavioral' : 'behavioral', 'Biological': 'biological', 'Combination Product': 'combination_product', 
        'Device': 'device', 'Diagnostic Test': 'diagnostic_test', 'Dietary Supplement': 'dietary_supplement',
        'Drug': 'drug', 'Genetic': 'genetic', 'Others': 'other', 'Procedure': 'procedure', 'Radiation': 'radiation'
    }

    condt_map = {
        'None': None, 'Cancer': 'cancers', 'Cardiovascular Disorder': 'cardiovascular_diseases',
        'Dental Disorder': 'dental_disorders', 'Dermatological Disorder': 'dermatological_disorders',
        'Endocrine/Metabolic Disorder': 'endocrine/metabolic_disorders',
        'Gastrointestinal Disorder': 'gastrointestinal_disorders', 'Genetic Disorder': 'genetic_disorders',
        'Infectious Disease': 'infectious_diseases', 'Mental Disorder': 'mental_disorders', 
        'Musculoskeletal Disorder': 'musculoskeletal_disorders', 'Neurological Disorder': 'neurological_disorders',
        'Ophthalmological Disorder': 'ophthalmological_disorders', 
        'Others': 'others', 'Pain Disorder': 'pain_disorders', 'Renal/Urological Disorder': 'renal/urological_disorders',
        'Reproductive Health': 'reproductive_health', 'Respiratory Disorder': 'respiratory_disorders'
    }

    allo_map = {'None': None, 'Non-Randomized': 'non_randomized', 'Randomized': 'randomized', 'Unknown': 'unknown'}

    masking_map = {'None': None, 'Double' : 'double', 'Quadruple' : 'quadruple', 'Single': 'single', 'Triple' : 'triple', 'Unknown': 'unknown'}

    model_map = {
        'None': None, 'Crossover' : 'crossover', 'Factorial' : 'factorial', 'Parallel' : 'parallel', 
        'Sequential': 'sequential', 'Single Group': 'single_group', 'Unknown': 'unknown'
    } 
        
    purpose_map = {
        'None': None, 'Diagnostic': 'diagnostic', 'Others': 'other', 'Prevention': 'prevention', 
        'Research': 'research', 'Supportive Care':  'supportive_care', 'Treatment' : 'treatment'
    }

    sponsor_map = {'Other': 'Other', 'Industry': 'Industry', 'Multiple': 'Multiple', 'Government': 'Government', 'Unknown' : 'Unknown'}

    age_map = {'Adult': 'adult', 'Child': 'child', 'Senior': 'senior', 'Mixed': 'mixed', 'Unknown' : 'Unknown'}

    gender_map = {'All': 'all', 'Male': 'male', 'Female': 'female'}

    binary_flags = [
        "has_dmc", "has_expanded_access", "is_fda_regulated_drug",
        "is_fda_regulated_device", "healthy_volunteers"
    ]

    flag_map = {"No": 0, "Yes": 1, "Unknown": -1}
    invt_map = {"No": 0, "Yes": 1}

    # === Model Selection ===
    model_choice = st.radio("Choose Model:", ["Logistic Regression", "XGBoost"])
    user_input = {}

    # === User Input Widgets ===
    st.markdown("### Trial Design Details")

    # Phase and Intervention
    col1, col2 = st.columns(2)
    with col1: sel_phase = st.selectbox("Phase(임상단계)", sorted(list(phase_map.keys())))
    with col2: sel_int = st.selectbox("Intervention Type(중재유형)", sorted(list(intervention_map.keys())))

    # Condition and Allocation
    col3, col4 = st.columns(2)
    with col3: sel_cond = st.selectbox("Condition(질환군)", sorted(list(condt_map.keys())))
    with col4: sel_allo = st.selectbox("Allocation(배정방식)", sorted(list(allo_map.keys())))

    # Masking and Model
    col5, col6 = st.columns(2)
    with col5: sel_masking = st.selectbox("Masking(눈가림방법)", sorted(list(masking_map.keys())))
    with col6: sel_model = st.selectbox("Study Model(연구모델)", sorted(list(model_map.keys())))

    # Purpose and Sponsor
    col7, col8 = st.columns(2)
    with col7: sel_pur = st.selectbox("Study Purpose(연구목적)", sorted(list(purpose_map.keys())))
    with col8: sel_sponsor = st.selectbox("Sponsor Type(스폰서유형)", sorted(list(sponsor_map.keys())))

    # Age and Gender
    col9, col10 = st.columns(2)
    with col9: sel_age = st.selectbox("Age (연령)", sorted(list(age_map.keys())))
    with col10: sel_gender = st.selectbox("Gender (성별)", sorted(list(gender_map.keys())))

    # Numeric Inputs
    st.markdown("### Numerical Inputs")
    col11, col12 = st.columns(2)
    with col11:
        enrollment_raw = st.number_input("Enrollment (등록인원수)", min_value=0, max_value=200000, step=10)
    with col12:
        duration_days  = st.number_input("Duration (연구기간(일))", min_value=0, max_value=10000, step=10)

    user_input["log_enrollment"] = float(np.log1p(enrollment_raw))  
    user_input["log_duration"]   = float(np.log1p(duration_days))

    col13, col14 = st.columns(2)
    with col13:
        user_input["number_of_arms"] = st.number_input("Number of Arms(시험군/대조군의 수)", min_value=1, max_value=20, step=1)
    with col14:
        user_input["intervention_count"] = st.number_input("Intervention Count(중재 갯수)", min_value=1, max_value=10, step=1)

    # === Flag Inputs ===
    flag_ko_map = {
        "has_dmc": "데이터 모니터링 위원회(DMC) 여부",
        "has_expanded_access": "확대 적용 여부",
        "is_fda_regulated_drug": "FDA 규제 약물 여부",
        "is_fda_regulated_device": "FDA 규제 기기 여부",
        "healthy_volunteers": "건강한 지원자 참여 여부"
    }
    st.markdown("### Flags")
    for flag in binary_flags:
        # 딕셔너리에서 한글 이름을 가져오고, 없으면 기본 영어 이름을 씁니다.
        label_en = flag.replace("_", " ").capitalize()
        label_ko = flag_ko_map.get(flag, "")
        full_label = f"{label_en} ({label_ko})" if label_ko else label_en
        
        sel = st.selectbox(full_label, list(flag_map.keys()), key=flag)
        user_input[flag] = flag_map[sel]
        
    sel_invt_option = st.selectbox('Has Multiple Intervention Types(복합치료 여부)', list(invt_map.keys()))
    user_input["has_multiple_intervention_types"] = invt_map[sel_invt_option]

    # === Clinical Trial Design Validity Checker ===
    invalid = False
    invalid_reasons = []

    # Critical Invalid Rules (Block Prediction)
    if sel_allo == "Randomized" and sel_model == "Single Group":
        invalid = True
        invalid_reasons.append("Randomized allocation with Single Group model is invalid.")

    if sel_masking == "Quadruple" and sel_model == "Single Group":
        invalid = True
        invalid_reasons.append("Quadruple masking cannot be used with Single Group design.")

    if sel_model == "Parallel" and sel_allo == "Randomized" and user_input["number_of_arms"] == 1:
        invalid = True
        invalid_reasons.append("Parallel Randomized studies must have more than one arm.")

    # Display Validation Messages
    if invalid:
        st.error("⚠️ Clinically invalid trial design detected. Please review the issues below:")
        for r in invalid_reasons:
            st.markdown(f"- {r}")
        st.stop() 

    # SHAP 레이블 정리 함수
    def clean_labels(explanation, input_row):
        explanation.feature_names = [
            f"{feat} ({round(input_row[feat], 2)})" if isinstance(input_row[feat], (int, float))
            else f"{feat} ({input_row[feat]})"
            for feat in explanation.feature_names
        ]
        return explanation

    # === Prediction ===
    if st.button("Predict Outcome"):
        if model_choice == "Logistic Regression":
            # Reset all one-hot columns
            for col in X_cols_logreg:
                if col.startswith(("phase_", "intervention_", "condt_", "allocation_", "masking_", "model_", "purpose_", "sponsor_", "elig_age_", "elig_gender_")):
                    user_input[col] = 0

            # Set selected categorical values (one-hot encoding)
            for sel, mapping, prefix in [
                (sel_phase, phase_map, ""),
                (sel_int, intervention_map, "intervention_"),
                (sel_cond, condt_map, "condt_"),
                (sel_allo, allo_map, "allocation_"),
                (sel_masking, masking_map, "masking_"),
                (sel_model, model_map, "model_"),
                (sel_pur, purpose_map, "purpose_"),
                (sel_sponsor, sponsor_map, "sponsor_"),
                (sel_age, age_map, "elig_age_"),
                (sel_gender, gender_map, "elig_gender_"),
            ]:
                mapped_val = mapping[sel]
                if mapped_val is not None:
                    col_name = f"{prefix}{mapped_val}"
                    if col_name in X_cols_logreg:
                        user_input[col_name] = 1

            input_df = pd.DataFrame([user_input]).reindex(columns=X_cols_logreg, fill_value=0)
            proba = logreg_model.predict_proba(input_df)[0][1]
            prediction = 1 if proba >= best_thr_lr else 0

        else:  # XGBoost
            user_input["phase_grouped"] = sorted(list(phase_map.keys())).index(sel_phase)
            user_input["intervention_grouped"] = sorted(list(intervention_map.keys())).index(sel_int) 
            user_input["condt_grouped"] = sorted(list(condt_map.keys())).index(sel_cond) 
            user_input["allocation_grouped"] = sorted(list(allo_map.keys())).index(sel_allo) 
            user_input["masking_grouped"] = sorted(list(masking_map.keys())).index(sel_masking) 
            user_input["model_grouped"] = sorted(list(model_map.keys())).index(sel_model) 
            user_input["purpose_grouped"] = sorted(list(purpose_map.keys())).index(sel_pur) 
            user_input["sponsor_grouped"] = sorted(list(sponsor_map.keys())).index(sel_sponsor) 
            user_input["elig_age_grouped"] = sorted(list(age_map.keys())).index(sel_age) 
            user_input["elig_gender_grouped"] = sorted(list(gender_map.keys())).index(sel_gender) 

            input_df = pd.DataFrame([user_input]).reindex(columns=X_cols_xgb, fill_value=0)
            proba = xgb_model.predict_proba(input_df)[0][1]
            prediction = 1 if proba >= best_thr_xgb else 0


        # === Display Results ===
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success(f"✅ Predicted Outcome: Success (Confidence: {proba*100:.2f}%)")
        else:
            st.error(f"❌ Predicted Outcome: Failure (Success probability: {proba*100:.2f}%)")

        st.caption(f"Model Used: {model_choice} (Youden threshold applied)")

        # === [추가] SHAP Interpretation ===
        st.markdown("---")
        st.subheader(" AI Decision Factors (SHAP Interpretation)")
        st.write("Below are the key factors contributing to this specific prediction.")

        try:
            # 모델 선택에 따른 Explainer 결정
            current_explainer = explainer_lr if model_choice == "Logistic Regression" else explainer_xgb
            
            # SHAP 값 계산
            # input_df는 이미 모델 형식에 맞춰져 있으므로 그대로 사용
            shap_values = current_explainer(input_df)
            
            # 레이블 정리 (입력한 값을 그래프 옆에 표시)
            # XGBoost는 숫자로 변환된 값을 보여주므로 원본 raw 데이터를 써도 좋습니다.
            shap_values_clean = clean_labels(shap_values[0], input_df.iloc[0])

            # 그래프 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values_clean, show=False)
            plt.title(f"Feature Contributions for {model_choice}", fontsize=12)
            
            # 스트림릿에 그래프 표시
            st.pyplot(fig)
            
            st.info("💡 **How to read:** Red bars increase the success probability, while Blue bars decrease it.")

        except Exception as e:
            st.warning("SHAP interpretation is currently unavailable for this configuration.")
        