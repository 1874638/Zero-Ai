# 제로부터 만드는 미니 대화형 LLM (PyTorch)

이 리포는 다음 단계를 통해 아주 작은 GPT를 직접 구현하고, 사전학습 → SFT(지시응답) → 대화를 수행합니다.

## 1) 설치
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 데이터·토크나이저 준비
```bash
python prepare_data.py --vocab_size 32000
```
- Tiny Shakespeare 코퍼스를 내려받고 Byte-Level BPE 토크나이저를 학습합니다.
- 생성물:
  - artifacts/tokenizer.json
  - data/train_ids.pt, data/val_ids.pt

## 3) 사전학습(언어모델)
```bash
python train_lm.py \
  --block_size 256 --n_layer 6 --n_head 6 --n_embd 384 \
  --batch_size 64 --lr 3e-4 --max_steps 5000
```
- 체크포인트:
  - checkpoints/pretrain/best.pt, last.pt

팁: VRAM이 작다면 n_layer/head/embd 또는 batch_size를 줄이세요.

## 4) SFT(지시-응답 미세조정)
`sft_data.jsonl`의 예시 포맷을 그대로 더 채워넣으세요.
```bash
python train_sft.py \
  --pretrain_ckpt checkpoints/pretrain/best.pt \
  --sft_data sft_data.jsonl \
  --epochs 3 --batch_size 16
```
- 체크포인트: checkpoints/sft/epoch{N}.pt

## 5) 대화
```bash
python chat.py --ckpt checkpoints/sft/epoch3.pt
```
- `exit` 입력 시 종료.

## 데이터 포맷(SFT)
- JSONL 각 줄: {instruction, input, output}
- 내부적으로 아래와 같이 포맷됩니다:
```
[SYSTEM] You are a helpful assistant.
[USER] <instruction>\n<input>
[ASSISTANT] <output> [EOS]
```
- 학습 시 `[ASSISTANT]` 이후 토큰만 손실로 계산(프롬프트는 마스킹)합니다.

## 한계와 다음 단계
- 이 예제는 교육용 최소 구현입니다. 작은 데이터/모델로는 긴 맥락·정확도 한계가 큽니다.
- 개선:
  - 더 큰/다양한 코퍼스로 사전학습 스텝 수 증가
  - 모델 확장(n_layer/head/embd), block_size 확장
  - 학습 안정화: 학습률 스케줄러, EMA, AMP(bfloat16/fp16)
  - 대화 데이터 확장 및 품질 관리
  - 선호도 학습(DPO/RLHF), 안전성 필터링
  - 빠른 추론: kv-cache 최적화, Torch compile, FlashAttention 등

## 요구사항/리소스 가이드
- GPU 8GB 기준: 본 설정 그대로 또는 n_embd/레이어 축소 권장
- CPU만 사용 시 매우 느립니다. 파라미터·스텝 수를 더 줄이고 실험해 보세요.