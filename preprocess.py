# src/preprocess.py
# ============================================================
#  MP3 보컬 파일 → MIDI 변환
# ============================================================

import os
import librosa
import numpy as np
import pretty_midi

import src.config as cfg


def mp3_to_midi(mp3_path: str = cfg.MP3_INPUT,
                midi_path: str = cfg.MIDI_OUTPUT) -> None:
    """
    MP3 보컬 파일 → MIDI 변환 후 저장

    Parameters
    ----------
    mp3_path  : 입력 MP3 파일 경로
    midi_path : 출력 MIDI 파일 경로
    """
    os.makedirs(os.path.dirname(midi_path), exist_ok=True)

    # ── 오디오 로드 ──────────────────────────────────────────
    print(f"  오디오 로드 중: {mp3_path}")
    y, sr = librosa.load(mp3_path, sr=22050)

    # ── 피치 추출 (pYIN 알고리즘) ────────────────────────────
    print("  피치 추출 중 (pYIN)...")
    f0, voiced_flag, _ = librosa.pyin(
        y, sr=sr,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        hop_length=512
    )
    times = librosa.times_like(f0, sr=sr, hop_length=512)

    # ── MIDI 객체 생성 ───────────────────────────────────────
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    note_start = None
    note_pitch = None

    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        current_time = float(times[i])

        if voiced and freq and freq > 0:
            current_pitch = int(round(librosa.hz_to_midi(freq)))
            current_pitch = max(0, min(127, current_pitch))

            if note_start is None:
                # 새 노트 시작
                note_start = current_time
                note_pitch = current_pitch
            elif abs(current_pitch - note_pitch) > cfg.PITCH_TOLERANCE:
                # 음정 변화 → 이전 노트 저장 후 새 노트 시작
                end_time = current_time
                if end_time - note_start >= cfg.MIN_NOTE_DURATION:
                    inst.notes.append(pretty_midi.Note(
                        velocity=100, pitch=note_pitch,
                        start=note_start, end=end_time
                    ))
                note_start = current_time
                note_pitch = current_pitch
        else:
            # 무음 구간 → 진행 중인 노트 마감
            if note_start is not None:
                end_time = current_time
                if end_time - note_start >= cfg.MIN_NOTE_DURATION:
                    inst.notes.append(pretty_midi.Note(
                        velocity=100, pitch=note_pitch,
                        start=note_start, end=end_time
                    ))
                note_start = None
                note_pitch = None

    # ── 마지막 노트 처리 ─────────────────────────────────────
    if note_start is not None:
        end_time = float(times[-1])
        if end_time - note_start >= cfg.MIN_NOTE_DURATION:
            inst.notes.append(pretty_midi.Note(
                velocity=100, pitch=note_pitch,
                start=note_start, end=end_time
            ))

    # ── MIDI 저장 ────────────────────────────────────────────
    midi.instruments.append(inst)
    midi.write(midi_path)

    total_duration = max(n.end for n in inst.notes) if inst.notes else 0
    m, s = divmod(total_duration, 60)
    print(f"✅ MIDI 변환 완료! → {midi_path}")
    print(f"   총 노트 수   : {len(inst.notes)}개")
    print(f"   MIDI 총 길이 : {total_duration:.2f}초 ({int(m)}분 {s:.1f}초)")
