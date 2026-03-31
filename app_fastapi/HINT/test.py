import sys
import os

# 1. 경로 설정을 최상단으로 올립니다.
current_dir = os.path.dirname(os.path.abspath(__file__)) # 하위 HINT 폴더
parent_dir = os.path.dirname(current_dir)                # 최상위 dev/HINT 폴더

sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 2. 경로 설정이 끝난 후에 모듈을 불러옵니다.
try:
    from molecule_encode import smiles2mpnnfeature
    print("모듈 로드 성공!")
except ImportError as e:
    print(f"모듈 로드 실패: {e}")
    # 만약 위에서 실패하면 직접 참조 시도
    import molecule_encode
    smiles2mpnnfeature = molecule_encode.smiles2mpnnfeature

# 3. 테스트 코드 실행 (smiles2mpnnfeature 스펠링 확인!)
test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
feature = smiles2mpnnfeature(test_smiles)

print(f'변환 성공! 특징 벡터 타입: {type(feature)}')
if hasattr(feature, 'shape'):
    print(f'특징 벡터 크기: {feature.shape}')
else:
    print("변환된 데이터 리스트가 생성되었습니다.")

import torch

# 가중치 파일 경로 확인 (본인의 경로에 맞게 수정)
ckpt_path = "../save_model/phase_II.ckpt"

# 모델 로드 시도
try:
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    print("가중치 파일을 성공적으로 불러왔습니다!")
    # 가중치 안에 어떤 내용이 있는지 살짝 보기
    print(f"저장된 키 목록: {checkpoint.keys()}")
except Exception as e:
    print(f"가중치 로드 실패: {e}")