import torch
import os

# 디바이스 설정 및 재현성 확보
torch.manual_seed(0) 
device = torch.device("cpu") # GPU 환경이라면 "cuda:0"로 변경 가능

import warnings
warnings.filterwarnings("ignore")

# HINT 파이프라인 모듈 Им포트
from HINT.dataloader import csv_three_feature_2_dataloader, generate_admet_dataloader_lst
from HINT.molecule_encode import MPNN, ADMET 
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding
from HINT.model import HINTModel 

print("==================================================")
print(" HINT: Hierarchical Interaction Network (Toy Test) ")
print("==================================================")

if not os.path.exists("figure"):
    os.makedirs("figure")

# 1. 태스크 및 데이터셋 위치 지정
base_name = 'toy' ### 사용 가능 옵션: 'toy', 'phase_I', 'phase_II', 'phase_III', 'indication'
datafolder = "data"
train_file = os.path.join(datafolder, base_name + '_train.csv')
valid_file = os.path.join(datafolder, base_name + '_valid.csv')
test_file = os.path.join(datafolder, base_name + '_test.csv')

# 2. ADMET 및 MPNN 모듈 로드/사전학습
print("\n[1/4] Preparing MPNN & ADMET model...")
mpnn_model = MPNN(mpnn_hidden_size=50, mpnn_depth=3, device=device)
admet_model_path = "save_model/admet_model.ckpt"

if not os.path.exists(admet_model_path):
    print("      -> Pretraining ADMET model from scratch...")
    admet_dataloader_lst = generate_admet_dataloader_lst(batch_size=32)
    admet_trainloader_lst = [i[0] for i in admet_dataloader_lst]
    admet_testloader_lst = [i[1] for i in admet_dataloader_lst]
    
    admet_model = ADMET(molecule_encoder=mpnn_model, 
                        highway_num=2, device=device, 
                        epoch=3, lr=5e-4, weight_decay=0, save_name='admet_')
    admet_model.train(admet_trainloader_lst, admet_testloader_lst)
    
    os.makedirs(os.path.dirname(admet_model_path), exist_ok=True)
    torch.save(admet_model, admet_model_path)
    print("      -> Saved ADMET model.")
else:
    print("      -> Loading pre-trained ADMET model...")
    admet_model = torch.load(admet_model_path, map_location=device)
    admet_model = admet_model.to(device)
    admet_model.set_device(device)

# 3. 데이터 로더 준비
print("\n[2/4] Loading Target Datasets...")
train_loader = csv_three_feature_2_dataloader(train_file, shuffle=True, batch_size=32) 
valid_loader = csv_three_feature_2_dataloader(valid_file, shuffle=False, batch_size=32) 
test_loader = csv_three_feature_2_dataloader(test_file, shuffle=False, batch_size=32) 

# 4. 질병(ICD10) 및 프로토콜 인코더 초기화
print("\n[3/4] Initializing Encoders (ICD10 & Protocol)...")
icdcode2ancestor_dict = build_icdcode2ancestor_dict()
gram_model = GRAM(embedding_dim=50, icdcode2ancestor=icdcode2ancestor_dict, device=device)
protocol_model = Protocol_Embedding(output_dim=50, highway_num=3, device=device)

# 5. HINT 모델 훈련 및 결과 검증(Inference)
print("\n[4/4] Learning & Inference (HINT)...")
hint_model_path = "save_model/" + base_name + ".ckpt"

# 모델 인스턴스화
model = HINTModel(molecule_encoder=mpnn_model, 
         disease_encoder=gram_model, 
         protocol_encoder=protocol_model,
         device=device, 
         global_embed_size=50, 
         highway_num_layer=2,
         prefix_name=base_name, 
         gnn_hidden_size=50,  
         epoch=3,
         lr=1e-3, 
         weight_decay=0)

if not os.path.exists(hint_model_path):
    print(f"      -> Training HINT '{base_name}' model from scratch...")
    model.init_pretrain(admet_model)
    model.learn(train_loader, valid_loader, test_loader)
    print("      -> Evaluating trained HINT model...")
    model.bootstrap_test(test_loader)
    torch.save(model, hint_model_path)
else:
    print(f"      -> Loading pre-trained HINT '{base_name}' model...")
    model = torch.load(hint_model_path, map_location=device)
    print("      -> Evaluating loaded HINT model on Test set...")
    model.bootstrap_test(test_loader)

print("\n==================================================")
print(" Process Complete. Results saved in `results/` folder. ")
print("==================================================")
