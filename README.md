%%writefile README.md
# 🎵 Vocal to Piano

보컬 멜로디를 입력받아 자동으로 피아노 반주를 생성하는 AI 모델입니다.  
Transformer 기반 Seq2Seq 구조를 사용하며, 학습된 가중치는 HuggingFace에서 자동으로 다운로드됩니다.

---

## 🔗 모델 가중치

| 항목 | 링크 |
|---|---|
| HuggingFace 모델 | [dksxodls1/vocal-to-piano](https://huggingface.co/dksxodls1/vocal-to-piano) |

> 별도 다운로드 불필요 — `generate.py` 실행 시 자동으로 다운로드됩니다.

---

## 📁 프로젝트 구조

```
vocal-to-piano/
├── src/
│   ├── config.py        # 하이퍼파라미터 설정
│   ├── vocab.py         # 토큰 사전 (Vocab)
│   ├── model.py         # Transformer 모델 구조
│   ├── dataset.py       # PyTorch Dataset / DataLoader
│   └── utils.py         # MIDI 처리, BPM 감지 등 유틸 함수
├── preprocess.py        # 데이터 전처리
├── train.py             # 모델 학습
├── generate.py          # 추론 (보컬 → 피아노 생성)
├── postprocess.py       # 후처리 (MIDI 정제)
├── requirements.txt     # 필요 패키지 목록
├── .gitignore           # Git 제외 파일 목록
└── README.md
```

---

## ⚙️ 설치 방법

```bash
git clone https://github.com/dksxodls1/vocal-to-piano.git
cd vocal-to-piano
pip install -r requirements.txt
```

---

## 🚀 사용 방법

### 1️⃣ 추론 실행 (보컬 → 피아노)

```bash
python generate.py --input vocal.mid --output output.mid
```

### 2️⃣ 직접 가중치 경로 지정

```bash
python generate.py --input vocal.mid --output output.mid --checkpoint best_model.pt
```

### 3️⃣ CLI 인자 설명

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--input` | 필수 | 입력 보컬 파일 경로 (`.mid`) |
| `--output` | `output.mid` | 출력 MIDI 파일 경로 |
| `--checkpoint` | `best_model.pt` | 가중치 경로 (없으면 자동 다운로드) |

---

## 🧠 모델 구조

```
[보컬 MIDI 입력]
      │
      ▼
[토큰화 (Vocab)]
      │
      ▼
[Transformer Encoder]
      │
      ▼
[Transformer Decoder]
      │
      ▼
[Top-k 샘플링 (Temperature)]
      │
      ▼
[피아노 화음 시퀀스 출력]
      │
      ▼
[MIDI 파일 저장]
```

---

## 🔧 주요 하이퍼파라미터

| 파라미터 | 설명 |
|---|---|
| `temperature` | 샘플링 다양성 조절 (높을수록 창의적) |
| `top_k` | 상위 k개 토큰 중 샘플링 |
| `note_div` | 음표 분할 단위 |
| `note_subdiv` | 음표 세부 분할 단위 |
| `d_model` | 모델 임베딩 차원 |
| `nhead` | Attention 헤드 수 |
| `num_layers` | Transformer 레이어 수 |
| `dropout` | 드롭아웃 비율 |
| `max_gen_len` | 최대 생성 시퀀스 길이 |

---

## 🔄 학습 방법

```bash
# 1. 데이터 전처리
python preprocess.py

# 2. 모델 학습
python train.py

# 3. 후처리
python postprocess.py
```

---

## 🗂️ 각 모듈 설명

| 파일 | 역할 |
|---|---|
| `src/config.py` | 학습 및 추론에 사용되는 모든 하이퍼파라미터 관리 |
| `src/vocab.py` | 화음/음표 토큰과 인덱스 간 변환 사전 |
| `src/model.py` | Transformer Encoder-Decoder 모델 정의 |
| `src/dataset.py` | PyTorch `Dataset` / `DataLoader` 구현 |
| `src/utils.py` | MIDI 파싱, BPM 감지, 시퀀스 변환 유틸 함수 |
| `preprocess.py` | 원시 MIDI 데이터를 토큰 시퀀스로 변환 |
| `train.py` | 모델 학습 루프 및 체크포인트 저장 |
| `generate.py` | 학습된 모델로 피아노 반주 생성 |
| `postprocess.py` | 생성된 MIDI 후처리 및 정제 |

---

## 📦 환경 정보

- Python 3.8+
- PyTorch 2.0+
- Google Colab (학습 환경)

---

## 📝 License

MIT License
