# train.py
# ============================================================
#  Vocal → Piano Accompaniment | 학습 스크립트
#  환경: Google Colab + A100 GPU
# ============================================================

import os
import csv
import math
import time
import random
import zipfile
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# ── 내부 모듈 import ─────────────────────────────────────────
import src.config as cfg
from src.vocab   import VOCAB_SIZE, PAD_IDX
from src.model   import VocalToPianoTransformer
from src.dataset import load_pop909_pairs, get_dataloaders


# ── 1. 재현성 고정 ───────────────────────────────────────────
def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 2. 압축 해제 ─────────────────────────────────────────────
def extract_zip(zip_path: str, out_dir: str) -> str:
    """ZIP 압축 해제 후 POP909 루트 경로 반환"""
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"  이미 압축 해제된 데이터 감지: {out_dir}")
    else:
        print(f"  압축 해제 중: {zip_path} -> {out_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        print("  압축 해제 완료!")

    candidates = [
        os.path.join(out_dir, "POP909-Dataset-master", "POP909"),
        os.path.join(out_dir, "POP909-Dataset-master"),
        os.path.join(out_dir, "POP909"),
        out_dir,
    ]
    for path in candidates:
        if os.path.isdir(path):
            subdirs = [d for d in os.listdir(path)
                       if os.path.isdir(os.path.join(path, d)) and d.isdigit()]
            if subdirs:
                print(f"  POP909 루트 감지: {path} ({len(subdirs)}개 곡 폴더)")
                return path

    for root, dirs, files in os.walk(out_dir):
        mid_files = [f for f in files if f.endswith('.mid') and f[:-4].isdigit()]
        if mid_files:
            parent = os.path.dirname(root)
            print(f"  POP909 루트 감지 (폴백): {parent}")
            return parent

    raise FileNotFoundError(
        f"POP909 곡 폴더를 찾을 수 없습니다. {out_dir} 내부를 확인하세요."
    )


# ── 3. LR 스케줄러 ───────────────────────────────────────────
class WarmupCosineScheduler:
    """Warm-up + Cosine Decay 학습률 스케줄러"""

    def __init__(self, optimizer, warmup_steps, total_steps, base_lr):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.base_lr      = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        s = self.current_step
        if s <= self.warmup_steps:
            lr = self.base_lr * (s / self.warmup_steps)
        else:
            progress = (s - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ── 4. 학습/평가 루프 ────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, device, epoch) -> float:
    model.train()
    total_loss, total_tokens = 0.0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [Train]", unit="batch")

    for src, tgt_in, tgt_out in pbar:
        src, tgt_in, tgt_out = (
            src.to(device), tgt_in.to(device), tgt_out.to(device)
        )
        logits = model(src, tgt_in)
        loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
        optimizer.step()
        lr = scheduler.step()

        n_tokens     = (tgt_out != PAD_IDX).sum().item()
        total_loss  += loss.item() * n_tokens
        total_tokens += n_tokens
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{lr:.2e}"})

    return total_loss / total_tokens if total_tokens > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d} [ Val ]", unit="batch")

    for src, tgt_in, tgt_out in pbar:
        src, tgt_in, tgt_out = (
            src.to(device), tgt_in.to(device), tgt_out.to(device)
        )
        logits = model(src, tgt_in)
        loss   = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

        n_tokens     = (tgt_out != PAD_IDX).sum().item()
        total_loss  += loss.item() * n_tokens
        total_tokens += n_tokens
        pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch'            : epoch,
        'model_state_dict' : model.state_dict(),
        'optim_state_dict' : optimizer.state_dict(),
        'val_loss'         : val_loss,
        'vocab_size'       : VOCAB_SIZE,
        'dataset'          : 'POP909',
    }, path)
    print(f"  [CKPT] 저장: {path}  (val_loss={val_loss:.4f})")


# ── 5. 메인 실행 ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  Vocal -> Piano Accompaniment | POP909 학습 시작")
    print("=" * 60)

    set_seed()
    print(f"  디바이스: {cfg.DEVICE} | 어휘 크기: {VOCAB_SIZE}")

    # Step 1: 데이터 준비
    print("\n[1/5] 데이터 준비 중...")
    pop909_root = extract_zip(cfg.ZIP_PATH, cfg.DATA_DIR)
    pairs = load_pop909_pairs(pop909_root)

    if not pairs:
        raise RuntimeError(
            f"유효한 POP909 파일 쌍을 찾을 수 없습니다: {cfg.DATA_DIR}"
        )

    # Step 2: DataLoader 구성
    print("\n[2/5] DataLoader 구성 중...")
    train_loader, val_loader = get_dataloaders(pairs)

    # Step 3: 모델 생성
    print("\n[3/5] 모델 생성 중...")
    model = VocalToPianoTransformer().to(cfg.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  모델 파라미터 수: {total_params:,}")

    # Step 4: 손실함수 / 옵티마이저 / 스케줄러
    print("\n[4/5] 옵티마이저 및 스케줄러 설정 중...")
    criterion   = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer   = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR,
        betas=(0.9, 0.98), weight_decay=1e-4
    )
    total_steps = len(train_loader) * cfg.EPOCHS
    scheduler   = WarmupCosineScheduler(
        optimizer, cfg.WARMUP, total_steps, cfg.LR
    )

    # Step 5: 학습 루프
    print(f"\n[5/5] 학습 루프 시작 (총 {cfg.EPOCHS} 에포크)")
    print("-" * 60)

    best_val_loss = float('inf')
    history       = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, cfg.DEVICE, epoch
        )
        val_loss = evaluate(model, val_loader, criterion, cfg.DEVICE, epoch)

        elapsed = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"\n  Epoch {epoch:02d}/{cfg.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"시간: {elapsed:.1f}s")

        # 최고 성능 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(cfg.CKPT_DIR, "best_model.pt")
            )

        # 매 5 에포크마다 중간 체크포인트 저장
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(cfg.CKPT_DIR, f"ckpt_epoch_{epoch:02d}.pt")
            )

        print("-" * 60)

    # 학습 완료 요약
    print("\n" + "=" * 60)
    print(f"  [DONE] 학습 완료! 최고 Val Loss: {best_val_loss:.4f}")
    print(f"  체크포인트: {cfg.CKPT_DIR}/best_model.pt")
    print("=" * 60)

    # 손실 기록 CSV 저장
    log_path = os.path.join(cfg.CKPT_DIR, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for i, (tl, vl) in enumerate(
                zip(history['train_loss'], history['val_loss']), 1):
            writer.writerow([i, f"{tl:.6f}", f"{vl:.6f}"])
    print(f"  학습 기록: {log_path}")


if __name__ == "__main__":
    main()
