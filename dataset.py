# src/dataset.py
# ============================================================
#  POP909 Dataset 클래스 및 DataLoader 생성 함수
# ============================================================

import os
import glob
import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

import src.config as cfg
from src.vocab import REST_IDX, SOS_IDX, EOS_IDX, PAD_IDX, QUALITIES
from src.utils import midi_track_to_chord_sequence


# ── 1. 품질 지표 계산 ────────────────────────────────────────
def compute_quality_metrics(vocal_seq: list, piano_seq: list) -> dict:
    """
    보컬-피아노 쌍의 품질 지표 계산

    Returns
    -------
    dict:
        activity_rate : 피아노 활성화율 (REST가 아닌 프레임 비율)
        harmony_score : 화성 일치율 (두 시퀀스 간 ROOT 일치율)
    """
    min_len = min(len(vocal_seq), len(piano_seq))
    if min_len == 0:
        return {'activity_rate': 0.0, 'harmony_score': 0.0}

    vocal_arr = np.array(vocal_seq[:min_len])
    piano_arr = np.array(piano_seq[:min_len])

    activity_rate = np.mean(piano_arr != REST_IDX)

    both_active = (vocal_arr != REST_IDX) & (piano_arr != REST_IDX)
    if both_active.sum() == 0:
        harmony_score = 0.0
    else:
        def get_root(idx_arr):
            shifted = idx_arr - 4
            return np.where(shifted >= 0, shifted // len(QUALITIES), -1)

        vocal_roots = get_root(vocal_arr[both_active])
        piano_roots = get_root(piano_arr[both_active])
        harmony_score = float(np.mean(vocal_roots == piano_roots))

    return {
        'activity_rate': float(activity_rate),
        'harmony_score': float(harmony_score),
    }


# ── 2. POP909 데이터 로드 ────────────────────────────────────
def load_pop909_pairs(
    pop909_root:       str,
    max_songs:         int   = cfg.MAX_SONGS,
    min_seq_len:       int   = cfg.SEQ_LEN,
    min_activity_rate: float = cfg.MIN_ACTIVITY_RATE,
    min_harmony_score: float = cfg.MIN_HARMONY_SCORE,
    use_transposition: bool  = cfg.USE_TRANSPOSITION,
    transpose_range:   list  = None,
) -> list:
    """
    POP909 실제 구조 기반 데이터 로더
    MELODY 트랙(0) → 보컬 화음 시퀀스 (src)
    PIANO  트랙(2) → 피아노 화음 시퀀스 (tgt)
    """
    if transpose_range is None:
        transpose_range = cfg.TRANSPOSE_RANGE

    song_dirs = sorted([
        d for d in glob.glob(os.path.join(pop909_root, "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ])[:max_songs]

    if len(song_dirs) == 0:
        raise FileNotFoundError(
            f"POP909 곡 폴더를 찾을 수 없습니다: {pop909_root}"
        )

    pairs = []
    stats = {
        'total': len(song_dirs), 'passed': 0,
        'missing_file': 0, 'too_short': 0,
        'low_quality': 0, 'parse_error': 0, 'augmented': 0,
    }

    for song_dir in tqdm(song_dirs, desc="  POP909 로딩", unit="곡"):
        song_id   = os.path.basename(song_dir)
        midi_path = os.path.join(song_dir, f"{song_id}.mid")

        if not os.path.exists(midi_path):
            stats['missing_file'] += 1
            continue

        try:
            midi_obj = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            stats['parse_error'] += 1
            continue

        if len(midi_obj.instruments) <= cfg.PIANO_TRACK_IDX:
            stats['missing_file'] += 1
            continue

        # 트랙명 검증 (경고만 출력)
        melody_name = midi_obj.instruments[cfg.MELODY_TRACK_IDX].name.upper()
        piano_name  = midi_obj.instruments[cfg.PIANO_TRACK_IDX].name.upper()
        if 'MELODY' not in melody_name or 'PIANO' not in piano_name:
            print(f"  [WARN] {song_id}: 트랙명 확인 필요 "
                  f"(트랙0={melody_name}, 트랙2={piano_name})")

        shifts = [0] + (transpose_range if use_transposition else [])

        for shift in shifts:
            try:
                vocal_seq = midi_track_to_chord_sequence(
                    midi_obj, cfg.MELODY_TRACK_IDX, shift
                )
                piano_seq = midi_track_to_chord_sequence(
                    midi_obj, cfg.PIANO_TRACK_IDX, shift
                )
            except Exception:
                if shift == 0:
                    stats['parse_error'] += 1
                continue

            if len(vocal_seq) < min_seq_len or len(piano_seq) < min_seq_len:
                if shift == 0:
                    stats['too_short'] += 1
                continue

            metrics = compute_quality_metrics(vocal_seq, piano_seq)
            if (metrics['activity_rate'] < min_activity_rate or
                    metrics['harmony_score'] < min_harmony_score):
                if shift == 0:
                    stats['low_quality'] += 1
                continue

            pairs.append((vocal_seq, piano_seq))
            if shift == 0:
                stats['passed'] += 1
            else:
                stats['augmented'] += 1

    # 로딩 결과 출력
    print(f"\n  {'='*45}")
    print(f"  POP909 로딩 결과")
    print(f"  {'='*45}")
    print(f"  전체 곡      : {stats['total']:>4}곡")
    print(f"  원본 통과    : {stats['passed']:>4}곡")
    print(f"  전조 증강    : {stats['augmented']:>4}쌍")
    print(f"  최종 쌍 수   : {len(pairs):>4}쌍")
    print(f"  파일 없음    : {stats['missing_file']:>4}곡")
    print(f"  시퀀스 짧음  : {stats['too_short']:>4}곡")
    print(f"  품질 미달    : {stats['low_quality']:>4}곡")
    print(f"  파싱 오류    : {stats['parse_error']:>4}곡")
    print(f"  {'='*45}\n")
    return pairs


# ── 3. Dataset 클래스 ────────────────────────────────────────
class VocalPianoDataset(Dataset):
    """
    슬라이딩 윈도우로 (src, tgt_in, tgt_out) 샘플 생성
    - src     : 보컬 화음 시퀀스 [SEQ_LEN]
    - tgt_in  : [SOS] + 피아노 시퀀스  (디코더 입력 / Teacher Forcing)
    - tgt_out : 피아노 시퀀스 + [EOS]  (정답 레이블)
    """

    def __init__(self, pairs: list):
        self.samples = []
        for vocal_seq, piano_seq in pairs:
            min_len = min(len(vocal_seq), len(piano_seq))
            for start in range(0, min_len - cfg.SEQ_LEN, cfg.STEP):
                src     = vocal_seq[start : start + cfg.SEQ_LEN]
                tgt     = piano_seq[start : start + cfg.SEQ_LEN]
                tgt_in  = [SOS_IDX] + tgt
                tgt_out = tgt + [EOS_IDX]
                self.samples.append((
                    torch.tensor(src,     dtype=torch.long),
                    torch.tensor(tgt_in,  dtype=torch.long),
                    torch.tensor(tgt_out, dtype=torch.long),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── 4. DataLoader 생성 함수 ──────────────────────────────────
def get_dataloaders(pairs: list):
    """
    VocalPianoDataset → train / val DataLoader 분리 후 반환
    """
    full_dataset = VocalPianoDataset(pairs)
    val_size     = int(len(full_dataset) * cfg.VAL_RATIO)
    train_size   = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"  슬라이딩 윈도우 샘플 : {len(full_dataset):,}개")
    print(f"  학습 샘플 : {len(train_ds):,}개 | 검증 샘플 : {len(val_ds):,}개")
    print(f"  배치 수   : {len(train_loader):,}개 (train) / {len(val_loader):,}개 (val)")

    return train_loader, val_loader
