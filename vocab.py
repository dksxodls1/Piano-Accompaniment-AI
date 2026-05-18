# src/vocab.py
# ============================================================
#  화음 어휘(Chord Vocabulary) 정의
#  모든 파일에서 공통으로 사용하는 화음 매핑 테이블
# ============================================================

ROOTS      = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
QUALITIES  = ['maj','min','dim','dom7']
CHORD_LIST = ['<PAD>', '<SOS>', '<EOS>', '<REST>'] + \
             [f"{r}_{q}" for r in ROOTS for q in QUALITIES]

CHORD2IDX  = {c: i for i, c in enumerate(CHORD_LIST)}
IDX2CHORD  = {i: c for c, i in CHORD2IDX.items()}
VOCAB_SIZE = len(CHORD_LIST)   # 4 + 48 = 52

PAD_IDX  = CHORD2IDX['<PAD>']
SOS_IDX  = CHORD2IDX['<SOS>']
EOS_IDX  = CHORD2IDX['<EOS>']
REST_IDX = CHORD2IDX['<REST>']

# ── 화음 템플릿 (반음 인터벌) ─────────────────────────────────
CHORD_TEMPLATES = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    'dim':  [0, 3, 6],
    'dom7': [0, 4, 7, 10],
}

# ── MIDI 음높이 기준음 (C4 = 60) ────────────────────────────
PITCH_BASE = {
    'C' : 60, 'C#': 61, 'D' : 62, 'D#': 63,
    'E' : 64, 'F' : 65, 'F#': 66, 'G' : 67,
    'G#': 68, 'A' : 69, 'A#': 70, 'B' : 71,
}
