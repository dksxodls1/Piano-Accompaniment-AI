# postprocess.py
# ============================================================
#  MIDI → WAV → MP3 변환 및 보컬 + 피아노 믹싱
# ============================================================

import os
import subprocess
from pydub import AudioSegment
from midi2audio import FluidSynth

import src.config as cfg


def midi_to_mp3(midi_path: str = cfg.PIANO_OUTPUT,
                wav_path:  str = cfg.WAV_OUTPUT,
                mp3_path:  str = cfg.MP3_PIANO) -> None:
    """MIDI → WAV → MP3 변환"""
    os.makedirs(os.path.dirname(mp3_path), exist_ok=True)

    # Step 1. MIDI → WAV
    print("  MIDI → WAV 변환 중...")
    FluidSynth().midi_to_audio(midi_path, wav_path)
    print(f"  WAV 변환 완료: {wav_path}")

    # Step 2. WAV → MP3
    print("  WAV → MP3 변환 중...")
    subprocess.run([
        "ffmpeg", "-i", wav_path, "-q:a", "2",
        mp3_path, "-y", "-loglevel", "quiet"
    ], check=True)
    print(f"  MP3 변환 완료: {mp3_path}")


def mix_vocal_piano(vocal_mp3:   str   = cfg.MP3_INPUT,
                    piano_mp3:   str   = cfg.MP3_PIANO,
                    output_mp3:  str   = cfg.MIXED_OUTPUT,
                    vocal_db:    float = -9.0,
                    piano_db:    float = +9.0) -> None:
    """
    보컬 MP3 + 피아노 MP3 믹싱 후 저장

    Parameters
    ----------
    vocal_db : 보컬 볼륨 조절 (dB)
    piano_db : 피아노 볼륨 조절 (dB)
    """
    os.makedirs(os.path.dirname(output_mp3), exist_ok=True)

    vocal = AudioSegment.from_mp3(vocal_mp3)
    piano = AudioSegment.from_mp3(piano_mp3)

    # 길이 맞추기 (짧은 쪽을 무음으로 패딩)
    if len(vocal) > len(piano):
        piano = piano + AudioSegment.silent(duration=len(vocal) - len(piano))
    else:
        vocal = vocal + AudioSegment.silent(duration=len(piano) - len(vocal))

    # 볼륨 조절
    vocal = vocal + vocal_db
    piano = piano + piano_db

    # 믹싱 및 저장
    mixed = vocal.overlay(piano)
    mixed.export(output_mp3, format="mp3", bitrate="320k")

    total_sec = len(mixed) / 1000
    m, s = divmod(total_sec, 60)
    print(f"✅ 믹싱 완료! → {output_mp3}")
    print(f"   총 길이: {total_sec:.1f}초 ({int(m)}분 {s:.1f}초)")


if __name__ == "__main__":
    midi_to_mp3()
    mix_vocal_piano()
