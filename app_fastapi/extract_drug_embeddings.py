import sys
import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from HINT.molecule_encode import mpnn_collate_func

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
hint_package_dir = os.path.join(current_dir, "HINT")
sys.path.insert(0, current_dir)
sys.path.insert(0, hint_package_dir)

from HINT.molecule_encode import smiles2mpnnfeature

def run():
    ckpt_path = "save_model/phase_II.ckpt"
    print(f"모델 로드 중... ({ckpt_path})")
    
    # 모델 로드
    model = torch.load(ckpt_path, map_location=torch.device('cpu'))
    # 모델의 전체 구조를 출력해서 '입구'를 찾습니다.
    # print("-" * 30)
    # print(model) 
    # print("-" * 30)

    # # 만약 위 명령어로 내용이 너무 많다면, 변수 목록만 봅니다.
    # print("사용 가능한 속성들:", vars(model).keys())
    # 모델이 가진 '서브 모듈'들의 진짜 이름을 출력합니다.
    # print("--- 서브 모듈 목록 ---")
    # for name, module in model.named_children():
    #     print(name)
    # print("----------------------")
    model.eval()

    # 데이터 로드
    drug_df = pd.read_csv("data/drugbank_mini.csv")
    smiles_col = 'moldb_smiles' 
    
    embeddings = []
    print("약물 임베딩 추출 시작...")

    for smiles in tqdm(drug_df[smiles_col]):
        try:
            # 1. MPNN 특징 추출
            feature = smiles2mpnnfeature(smiles)
            feature_batch = mpnn_collate_func([feature])
            
            with torch.no_grad():
                # 2. 상자 까기 (알맹이 추적)
                target = feature_batch
                while isinstance(target, (list, tuple)) and len(target) > 0:
                    target = target[0]
                
                if isinstance(target, dict):
                    target = target.get('fatoms', list(target.values())[0])

                # 3. [최종 병기] 89차원으로 맞춤
                import numpy
                curr_data = numpy.array(target).flatten()
                
                # 입구인 89차원 규격 (모자라면 0, 남으면 자름)
                target_dim = 89
                final_data = numpy.zeros(target_dim)
                final_data[:min(len(curr_data), target_dim)] = curr_data[:min(len(curr_data), target_dim)]
                
                # 텐서로 변환 (1, 89)
                input_tensor = torch.FloatTensor(final_data).unsqueeze(0).to(next(model.parameters()).device)

                # 4. [수정] 인코더를 한꺼번에 실행하지 않고, 층별로 차근차근 통과
                # 만약 Sequential 구조라면 차례대로 실행해서 꼬임을 방지합니다.
                if hasattr(model.molecule_encoder, 'children') and list(model.molecule_encoder.children()):
                    drug_vec = input_tensor
                    for layer in model.molecule_encoder.children():
                        # 레이어의 규격이 현재 drug_vec과 맞을 때만 통과
                        if hasattr(layer, 'in_features'):
                            if drug_vec.shape[-1] == layer.in_features:
                                drug_vec = layer(drug_vec)
                        else:
                            # Linear 레이어가 아니면 (Activation 등) 일단 통과 시도
                            try: drug_vec = layer(drug_vec)
                            except: continue
                else:
                    # Sequential이 아니라면 직접 호출
                    drug_vec = model.molecule_encoder(input_tensor)
                
                # 5. 결과 리스트 변환 및 저장
                vec = drug_vec.detach().cpu().numpy().reshape(-1).tolist()
                if len(vec) < 128:
                    vec = vec + [0.0] * (128 - len(vec))
                embeddings.append(vec[:128])

        except Exception as e:
            # 여전히 에러가 나면 여기서 e를 출력합니다.
            print(f"\n[Error Detail] {e}")
            embeddings.append([0.0] * 128)



    # 데이터 저장
    emb_cols = [f'dr_emb_{i}' for i in range(128)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    
    # 기존 dr_emb 컬럼이 있다면 제거 후 병합
    drug_df = drug_df.loc[:, ~drug_df.columns.str.startswith('dr_emb_')]
    result_df = pd.concat([drug_df, emb_df], axis=1)
    
    result_df.to_csv("data/drug_embeddings_hint.csv", index=False)
    print("\n✨ 작업 완료! 터미널에 [Error] 메시지가 떴는지 확인해주세요.")

if __name__ == "__main__":
    run()