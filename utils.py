# src/utils.py
# ============================================================
#  화음 유틸리티 함수 모음
#  - MIDI 피치 → 화음 인덱스 변환
#  - MIDI 객체 → 화음 시퀀스 변환
#  - 화음 시퀀스 → MIDI 저장
#  - BPM → TEMPO_RES 변환
#  - MIDI에서 BPM 자동 감지
# ============================================================

import numpy as np
import pretty_midi

import src.config as cfg
from src.vocab import (
    ROOTS, QUALITIES, CHORD_TEMPLATES, PITCH_BASE,
    CHORD2IDX, IDX2CHORD, REST_IDX
)


# ── 1. 피치 목록 → 화음 인덱스 ──────────────────────────────
def notes_to_chord_idx(active_pitches: list) -> int:
    """
    동시에 울리는 MIDI 피치 목록 → 가장 잘 맞는 화음 인덱스 반환

    Parameters
    ----------
    active_pitches : 동시에 울리는 MIDI 피치 번호 리스트

    Returns
    -------
    int : 화음 인덱스 (매칭 실패 시 REST_IDX 반환)
    """
    if not active_pitches:
        return REST_IDX

    pitch_classes = set(p % 12 for p in active_pitches)
    best_score, best_chord = -1, None

    for root_idx, root in enumerate(ROOTS):
        for quality, intervals in CHORD_TEMPLATES.items():
            template = set((root_idx + iv) % 12 for iv in intervals)
            score = len(pitch_classes & template) / len(template)
            if score > best_score:
                best_score = score
                best_chord = f"{root}_{quality}"

    return CHORD2IDX.get(best_chord, REST_IDX)


# ── 2. 단일 트랙 MIDI 파일 경로 → 화음 시퀀스 ────────────────
def midi_to_chord_sequence(midi_path: str,
                            semitone_shift: int = 0) -> list:
    """
    단일 트랙 MIDI 파일 경로 → 화음 인덱스 정수 리스트 변환
    generate.py에서 MP3→MIDI 변환 후 활용

    Parameters
    ----------
    midi_path      : MIDI 파일 경로 (단일 트랙)
    semitone_shift : 키 전조 반음 수 (0 = 원본)

    Returns
    -------
    list : 화음 인덱스 정수 리스트
    """
    try:
        midi_obj = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    if len(midi_obj.instruments) == 0:
        return []

    duration   = midi_obj.get_end_time()
    times      = np.arange(0, duration, cfg.TEMPO_RES)
    instrument = midi_obj.instruments[0]
    chord_seq  = []

    for t in times:
        active = [
            n.pitch + semitone_shift
            for n in instrument.notes
            if n.start <= t < n.end
        ]
        active = [max(0, min(127, p)) for p in active]
        chord_seq.append(notes_to_chord_idx(active))

    return chord_seq


# ── 3. PrettyMIDI 객체의 특정 트랙 → 화음 시퀀스 ────────────
def midi_track_to_chord_sequence(midi_obj: pretty_midi.PrettyMIDI,
                                  track_idx: int,
                                  semitone_shift: int = 0) -> list:
    """
    POP909용: PrettyMIDI 객체의 특정 트랙 → 화음 인덱스 리스트 변환

    Parameters
    ----------
    midi_obj       : 이미 로드된 PrettyMIDI 객체
    track_idx      : 사용할 트랙 인덱스
                     (0 = MELODY 보컬 멜로디 / 2 = PIANO 피아노 반주)
    semitone_shift : 키 전조 반음 수 (0 = 원본)

    Returns
    -------
    list : 화음 인덱스 정수 리스트
    """
    if track_idx >= len(midi_obj.instruments):
        return []

    instrument = midi_obj.instruments[track_idx]
    duration   = midi_obj.get_end_time()
    times      = np.arange(0, duration, cfg.TEMPO_RES)
    chord_seq  = []

    for t in times:
        active = [
            n.pitch + semitone_shift
            for n in instrument.notes
            if n.start <= t < n.end
        ]
        active = [max(0, min(127, p)) for p in active]
        chord_seq.append(notes_to_chord_idx(active))

    return chord_seq


# ── 4. 화음 시퀀스 → MIDI 파일 저장 ─────────────────────────
def chord_seq_to_midi(chord_indices: list,
                      output_path:   str,
                      tempo:         float = 120.0,
                      start_time:    float = 0.0,
                      note_subdiv:   int   = 1,
                      tempo_res:     float = None) -> None:
    """
    화음 인덱스 리스트 → MIDI 파일로 저장

    Parameters
    ----------
    chord_indices : 화음 인덱스 리스트
    output_path   : 저장할 MIDI 파일 경로
    tempo         : BPM
    start_time    : 무음 구간 오프셋 (초)
    note_subdiv   : TEMPO_RES 안에서 노트를 몇 개로 분할할지
                    (1 = 분할 없음, 2 = 절반씩 2개)
    tempo_res     : 프레임 하나의 길이 (None이면 cfg.TEMPO_RES 사용)
    """
    if tempo_res is None:
        tempo_res = cfg.TEMPO_RES

    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    note_dur = tempo_res / note_subdiv

    for i, idx in enumerate(chord_indices):
        chord_name = IDX2CHORD.get(idx, '<REST>')
        if chord_name in ('<PAD>', '<SOS>', '<EOS>', '<REST>'):
            continue
        try:
            root_str, quality = chord_name.split('_')
            root_pitch = PITCH_BASE[root_str]
            intervals  = CHORD_TEMPLATES[quality]

            for sub in range(note_subdiv):
                start = start_time + i * tempo_res + sub * note_dur
                end   = start + note_dur
                for interval in intervals:
                    piano.notes.append(pretty_midi.Note(
                        velocity=80,
                        pitch=root_pitch + interval,
                        start=start,
                        end=end,
                    ))
        except Exception:
            continue

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"  MIDI 저장 완료: {output_path}")


# ── 5. BPM → TEMPO_RES 변환 ─────────────────────────────────
def bpm_to_tempo_res(bpm: float, note_div: float) -> float:
    """
    BPM과 note_div로 TEMPO_RES(프레임 길이) 계산

    Parameters
    ----------
    bpm      : 템포 (BPM)
    note_div : 박자 단위
               (0.5 = 2분음표, 1.0 = 4분음표, 2.0 = 8분음표)

    Returns
    -------
    float : 프레임 하나의 길이 (초)

    Examples
    --------
    >>> bpm_to_tempo_res(120, 0.5)   # 2분음표 → 0.25초
    >>> bpm_to_tempo_res(120, 1.0)   # 4분음표 → 0.125초
    """
    tempo_res = 60.0 / (bpm * note_div)
    print(f"  note_div={note_div} → TEMPO_RES: {tempo_res:.4f}초")
    return tempo_res


# ── 6. MIDI에서 BPM 자동 감지 ───────────────────────────────
def detect_bpm_from_midi(midi_path: str) -> float:
    """
    MIDI 파일 내부 템포 이벤트에서 BPM 추출

    Parameters
    ----------
    midi_path : MIDI 파일 경로

    Returns
    -------
    float : BPM 값

    Raises
    ------
    ValueError : MIDI 파일에 템포 정보가 없을 경우
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    _, tempo_arr = midi.get_tempo_changes()
    tempo_arr = np.atleast_1d(tempo_arr)

    if len(tempo_arr) == 0:
        raise ValueError("MIDI 파일에 템포 정보가 없습니다.")

    return float(tempo_arr[0])
