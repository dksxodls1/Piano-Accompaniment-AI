%%writefile generate.py

import os
import torch
import argparse
import numpy as np
from huggingface_hub import hf_hub_download

from src.config import Config
from src.model import VocalToPianoModel
from src.vocab import Vocab
from src.utils import (
    midi_to_chord_sequence,
    chord_sequence_to_midi,
    detect_bpm,
)

# ─────────────────────────────────────────────
# HuggingFace 설정
# ─────────────────────────────────────────────
REPO_ID        = "dksxodls1/vocal-to-piano"
MODEL_FILENAME = "best_model.pt"


# ─────────────────────────────────────────────
# 1. 모델 가중치 자동 다운로드 함수
# ─────────────────────────────────────────────
def load_model_weights(local_path="best_model.pt"):
    if not os.path.isfile(local_path):
        print(f"📥 '{local_path}' 없음 — HuggingFace에서 다운로드 중...")
        local_path = hf_hub_download(
            repo_id  = REPO_ID,
            filename = MODEL_FILENAME,
        )
        print("✅ 다운로드 완료!")
    else:
        print(f"✅ 로컬 가중치 파일 확인: {local_path}")
    return local_path


# ─────────────────────────────────────────────
# 2. 모델 초기화 및 가중치 로드
# ─────────────────────────────────────────────
def load_model(cfg, vocab, device, checkpoint_path="best_model.pt"):
    model = VocalToPianoModel(
        vocab_size = vocab.size,
        d_model    = cfg.d_model,
        nhead      = cfg.nhead,
        num_layers = cfg.num_layers,
        dropout    = cfg.dropout,
    ).to(device)

    checkpoint_path = load_model_weights(checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"🎹 모델 로드 완료: {checkpoint_path}")
    return model


# ─────────────────────────────────────────────
# 3. 시퀀스 생성 함수
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_sequence(model, input_seq, vocab, cfg, device):
    model.eval()
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    generated = []
    decoder_input = torch.tensor(
        [[vocab.token_to_idx["<BOS>"]]], dtype=torch.long
    ).to(device)

    for _ in range(cfg.max_gen_len):
        output = model(input_tensor, decoder_input)
        logits = output[:, -1, :]

        logits = logits / cfg.temperature
        probs  = torch.softmax(logits, dim=-1)

        top_k_probs, top_k_idx = torch.topk(probs, cfg.top_k, dim=-1)
        next_token = top_k_idx[0][torch.multinomial(top_k_probs[0], 1)]

        token_str = vocab.idx_to_token[next_token.item()]

        if token_str == "<EOS>":
            break

        generated.append(next_token.item())
        decoder_input = torch.cat(
            [decoder_input, next_token.unsqueeze(0).unsqueeze(0)], dim=1
        )

    return generated


# ─────────────────────────────────────────────
# 4. 메인 실행
# ─────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    cfg   = Config()
    vocab = Vocab()

    model = load_model(cfg, vocab, device, checkpoint_path=args.checkpoint)

    print(f"🎤 입력 파일 처리 중: {args.input}")
    bpm = detect_bpm(args.input)
    print(f"🎵 감지된 BPM: {bpm}")

    input_seq = midi_to_chord_sequence(
        args.input,
        vocab       = vocab,
        note_div    = cfg.note_div,
        note_subdiv = cfg.note_subdiv,
        bpm         = bpm,
    )

    if len(input_seq) == 0:
        print("❌ 입력 시퀀스가 비어 있어! 파일 확인 필요.")
        return

    print("🎹 피아노 반주 생성 중...")
    generated_seq = generate_sequence(model, input_seq, vocab, cfg, device)

    if len(generated_seq) == 0:
        print("❌ 생성된 시퀀스가 비어 있어!")
        return

    chord_sequence_to_midi(
        generated_seq,
        vocab       = vocab,
        output_path = args.output,
        bpm         = bpm,
        note_div    = cfg.note_div,
        note_subdiv = cfg.note_subdiv,
    )
    print(f"✅ 생성 완료! 저장 위치: {args.output}")


# ─────────────────────────────────────────────
# 5. CLI 인자 파싱
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vocal to Piano Generator")

    parser.add_argument("--input",      type=str, required=True,              help="입력 보컬 파일 경로")
    parser.add_argument("--output",     type=str, default="output.mid",       help="출력 MIDI 파일 경로")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt",    help="모델 가중치 경로")

    main(parser.parse_args())
