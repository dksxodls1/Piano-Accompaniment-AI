# 🎹 Piano Accompaniment AI

보컬 MP3를 입력받아 자동으로 피아노 반주 MIDI를 생성하는 딥러닝 기반 AI 프로젝트입니다.

---

## 📌 주요 기능

- 🎤 보컬 MP3 → 피아노 반주 MIDI 자동 생성
- 🔇 무음 구간 감지 및 MIDI 출력에 반영 (`start_time` 처리)
- 🎼 화음 진행 다양성 조절 (`temperature`, `top_k`)
- 🎵 반주 재생 간격 및 코드 변경 주기 커스터마이징 (`note_div`, `note_subdiv`)

---

## 📁 디렉터리 구조

```
piano-accompaniment-ai/
│
├── data/
│   ├── raw/                  # 원본 MP3 파일
│   ├── processed/            # 전처리된 chord sequence 데이터
│   └── midi/                 # 생성된 MIDI 출력 파일
│
├── model/
│   ├── checkpoints/          # 학습된 모델 가중치 (.pt)
│   └── configs/              # 하이퍼파라미터 설정 파일
│
├── src/
│   ├── train.py              # 모델 학습 스크립트
│   ├── generate.py           # 반주 생성 스크립트
│   ├── preprocess.py         # MP3 전처리 (무음 구간 포함)
│   ├── dataset.py            # Dataset / DataLoader 정의
│   ├── model.py              # 모델 아키텍처
│   └── utils.py              # chord_seq_to_midi 등 유틸 함수
│
├── notebooks/
│   └── experiment.ipynb      # 실험 및 분석용 노트북
│
├── tests/
│   ├── test_model.py
│   └── test_generate.py
│
├── docs/
│   └── architecture.md       # 모델 구조 상세 설명
│
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ 설치

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/piano-accompaniment-ai.git
cd piano-accompaniment-ai
```

### 2. 가상환경 생성 (선택)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

---

## 🚀 사용법

### 📦 데이터 전처리

```bash
python src/preprocess.py --input_dir data/raw/ --output_dir data/processed/
```

### 🏋️ 모델 학습

```bash
python src/train.py \
  --data_dir data/processed/ \
  --checkpoint_dir model/checkpoints/ \
  --epochs 100 \
  --batch_size 32
```

### 🎹 반주 생성

```bash
python src/generate.py \
  --input vocals.mp3 \
  --output data/midi/output.mid \
  --temperature 0.85 \
  --top_k 7 \
  --note_div 2 \
  --note_subdiv 2
```

---

## 🎛️ 주요 파라미터

### 생성 품질 관련

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--temperature` | `0.85` | 화음 다양성 조절 (낮을수록 안정적, 높을수록 다양) |
| `--top_k` | `7` | 화음 후보 수 (낮을수록 보수적, 높을수록 창의적) |

### 반주 구조 관련

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--note_div` | `2` | 마디 분할 단위 (2 = 2분음표마다 음 배치) |
| `--note_subdiv` | `2` | 코드 변경 주기 (2 = note_div 2개마다 코드 교체) |

### `temperature` × `top_k` 조합 가이드

| temperature | top_k | 특징 |
|---|---|---|
| 0.6 ~ 0.7 | 3 ~ 5 | 매우 안정적, 단조로울 수 있음 |
| **0.85** | **7** | **균형 잡힌 추천 설정** ✅ |
| 1.0 | 10 | 다양하고 창의적, 불안정할 수 있음 |
| 1.2↑ | 15↑ | 예측 불가능, 실험용 |

---

## 🛠️ 알려진 버그 수정 내역

- ✅ 모델 로드 시 아키텍처 불일치 오류 수정
- ✅ `generate_full` 및 `chord_seq_to_midi` 중복 호출 버그 수정
- ✅ 입력 MP3의 무음 구간이 출력 MIDI에 반영되지 않던 문제 (`start_time` 전달로 해결)

---

## 📊 학습 결과

> 학습된 모델 가중치는 용량 문제로 저장소에 포함되지 않습니다.
> 아래 링크에서 다운로드 후 `model/checkpoints/`에 위치시켜 주세요.

- 📥 [Google Drive에서 다운로드](#) _(링크 추가 예정)_
- 🤗 [HuggingFace Hub](#) _(링크 추가 예정)_

---

## 📋 Requirements

```
torch>=2.0.0
torchaudio>=2.0.0
pretty_midi
librosa
numpy
tqdm
```

---

## 📄 라이선스

This project is licensed under the [MIT License](LICENSE).

---

## 🙋 문의

프로젝트 관련 문의는 [Issues](https://github.com/your-username/piano-accompaniment-ai/issues) 탭을 이용해 주세요.
