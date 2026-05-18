# src/config.py
# ============================================================
#  프로젝트 전역 설정 (모든 파일이 이 파일을 import)
# ============================================================

import os
import torch

# ── 경로 ────────────────────────────────────────────────────
ZIP_PATH = "/content/drive/MyDrive/POP909-Dataset-master.zip"
DATA_DIR = "/content/pop909_data"
CKPT_DIR = "/content/drive/MyDrive/vp_ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── 입출력 경로 ──────────────────────────────────────────────
MP3_INPUT    = "/content/drive/MyDrive/input/input_mp3/vocal.mp3"
MIDI_OUTPUT  = "/content/drive/MyDrive/input/mp3-midi/vocal_input.mid"
PIANO_OUTPUT = "/content/drive/MyDrive/background_output/piano_output.mid"
WAV_OUTPUT   = "/content/drive/MyDrive/background_output/piano_output.wav"
MP3_PIANO    = "/content/drive/MyDrive/background_output/piano_output.mp3"
MIXED_OUTPUT = "/content/drive/MyDrive/background_output/mixed_output.mp3"

# ── 데이터 ──────────────────────────────────────────────────
SEQ_LEN   = 128
STEP      = 64
TEMPO_RES = 0.125   # 초당 8 프레임 (0.125s 단위)
MAX_SONGS = 909

# ── POP909 트랙 인덱스 ───────────────────────────────────────
MELODY_TRACK_IDX = 0   # 트랙 0 = MELODY
PIANO_TRACK_IDX  = 2   # 트랙 2 = PIANO

# ── 데이터 증강 ──────────────────────────────────────────────
USE_TRANSPOSITION = True
TRANSPOSE_RANGE   = [-2, -1, 1, 2]

# ── 품질 필터 ────────────────────────────────────────────────
MIN_ACTIVITY_RATE = 0.15
MIN_HARMONY_SCORE = 0.25

# ── MP3→MIDI 전처리 ──────────────────────────────────────────
MIN_NOTE_DURATION = 0.05   # 50ms 미만 노트 무시
PITCH_TOLERANCE   = 1      # ±1 semitone 오차 허용

# ── 모델 하이퍼파라미터 ──────────────────────────────────────
D_MODEL  = 256
N_HEADS  = 4
N_LAYERS = 4
FFN_DIM  = 1024
DROPOUT  = 0.1

# ── 학습 하이퍼파라미터 ──────────────────────────────────────
SEED       = 42
EPOCHS     = 30
BATCH_SIZE = 128
LR         = 3e-4
WARMUP     = 1000
CLIP_GRAD  = 1.0
VAL_RATIO  = 0.1

# ── 디바이스 ────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
