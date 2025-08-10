프로젝트 준비
새 폴더에 제가 준 파일들을 그대로 저장하세요. 예: mini-llm/
가상환경 생성 후 의존성 설치
bash
python -m venv .venv
# Windows: .venv\Scripts\activate, mac/Linux: source .venv/bin/activate
source .venv/bin/activate
pip install -r requirements.txt
데이터·토크나이저 준비
Tiny Shakespeare 말뭉치를 내려받고 Byte-Level BPE 토크나이저를 학습합니다.
bash
python prepare_data.py --vocab_size 32000
생성물
artifacts/tokenizer.json
data/train_ids.pt, data/val_ids.pt
사전학습(언어모델)
작은 GPT를 언어모델로 먼저 학습합니다.
bash
python train_lm.py \
  --block_size 256 --n_layer 6 --n_head 6 --n_embd 384 \
  --batch_size 64 --lr 3e-4 --max_steps 5000
체크포인트: checkpoints/pretrain/best.pt, last.pt
메모리가 부족하면 n_layer/n_embd/batch_size를 줄이세요. 예: n_layer=4, n_embd=256, batch_size=16
SFT(지시-응답 성향 주입)
sft_data.jsonl을 열어서 동일 포맷으로 예시를 더 추가하세요. 각 줄은 독립적인 JSON 객체입니다.
키: instruction, input(없으면 빈 문자열), output
학습 실행
bash
python train_sft.py \
  --pretrain_ckpt checkpoints/pretrain/best.pt \
  --sft_data sft_data.jsonl \
  --epochs 3 --batch_size 16
체크포인트: checkpoints/sft/epoch{N}.pt
대화 실행
bash
python chat.py --ckpt checkpoints/sft/epoch3.pt
프롬프트가 뜨면 질문을 입력하세요. 종료: exit
자주 겪는 이슈/해결

CUDA 메모리 부족(OutOfMemory)
batch_size를 크게 줄이기, n_embd/ n_layer 감소, block_size 줄이기(예: 256→128)
GPU가 없을 때
모든 스크립트에 --device cpu 옵션을 추가하면 됩니다. 다만 매우 느립니다.
예: python train_lm.py --device cpu
JSONL 형식 오류
각 줄이 완전한 JSON이어야 합니다. 파일은 UTF-8로 저장하세요.
학습이 너무 느릴 때
max_steps를 줄이거나 batch_size 축소, 모델 크기 축소
디렉터리 구조 예시(정상 실행 후)

Code
mini-llm/
├─ requirements.txt
├─ prepare_data.py
├─ gpt.py
├─ train_lm.py
├─ sft_data.jsonl
├─ train_sft.py
├─ chat.py
├─ artifacts/
│  └─ tokenizer.json
├─ data/
│  ├─ train_ids.pt
│  └─ val_ids.pt
└─ checkpoints/
   ├─ pretrain/
   │  ├─ best.pt
   │  └─ last.pt
   └─ sft/
      └─ epoch3.pt
