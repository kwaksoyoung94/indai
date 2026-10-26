"""Stage 2 CNN-LSTM 학습. 영상 단위로 분할된 train/val 만 사용하며 모델 선택은 val loss 기준.

    python train.py --name cnn_lstm --backbone scratch
    python train.py --name mnv3 --backbone mobilenet_v3_small --pretrained
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from config import Config
from dataset.data_loader import get_loader
from models.cnn_lstm import CNNLSTM, R3D18, save_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot, n, probs, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(':')[0], dtype=torch.float16,
                                enabled=device.startswith('cuda')):
                logit = model(x)
                loss = loss_fn(logit.float(), y)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
            probs.append(torch.sigmoid(logit.float()).detach().cpu())
            ys.append(y.cpu())
    probs, ys = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    auc = float(roc_auc_score(ys, probs)) if len(set(ys.tolist())) == 2 else None
    acc = float(((probs >= 0.5) == (ys >= 0.5)).mean())
    return tot / max(n, 1), acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cnn_lstm')
    ap.add_argument('--arch', default='cnn_lstm', choices=['cnn_lstm', 'r3d_18'], help='비교용 3D CNN 은 r3d_18')
    ap.add_argument('--backbone', default=Config.BACKBONE)
    ap.add_argument('--pretrained', action='store_true')
    ap.add_argument('--epochs', type=int, default=Config.EPOCHS)
    ap.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    ap.add_argument('--seq-len', type=int, default=Config.SEQUENCE_LENGTH)
    ap.add_argument('--dropout', type=float, default=Config.DROPOUT)
    ap.add_argument('--wd', type=float, default=Config.WEIGHT_DECAY)
    ap.add_argument('--hidden', type=int, default=Config.LSTM_HIDDEN)
    ap.add_argument('--temporal', default='lstm', choices=['lstm', 'gru', 'transformer', 'mean'])
    ap.add_argument('--seed', type=int, default=Config.SEED)
    ap.add_argument('--fold', type=int, default=None, help='k-fold 교차검증의 test fold 번호')
    a = ap.parse_args()
    if a.fold is not None:
        a.name = f'{a.name}_fold{a.fold}'

    set_seed(a.seed)
    device = Config.DEVICE
    print(f'device={device}  {torch.cuda.get_device_name(0) if device == "cuda" else ""}')

    train_loader = get_loader('train', a.batch_size, a.seq_len, augment=True, shuffle=True, fold=a.fold)
    val_loader = get_loader('val', a.batch_size, a.seq_len, fold=a.fold)
    ytr = train_loader.dataset.df.y.values
    n_pos, n_neg = int((ytr == 1).sum()), int((ytr == 0).sum())
    print(f'train clips: fire={n_pos}  non-fire={n_neg}   val clips: {len(val_loader.dataset)}')
    if n_pos == 0 or n_neg == 0:
        raise SystemExit('train split에 한 클래스만 있습니다.')
    pos_weight = torch.tensor(n_neg / n_pos, device=device)   # 클래스 불균형 보정
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if a.arch == 'r3d_18':
        model = R3D18(a.pretrained, dropout=a.dropout).to(device)
    else:
        model = CNNLSTM(a.backbone, a.pretrained, hidden=a.hidden, dropout=a.dropout, temporal=a.temporal).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')

    out_dir = Config.WEIGHTS_DIR / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best, bad = [], float('inf'), 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, device, opt, scaler)
        va = run_epoch(model, val_loader, loss_fn, device)
        history.append(dict(epoch=ep, train_loss=tr[0], train_acc=tr[1], val_loss=va[0],
                            val_acc=va[1], val_auc=va[2], sec=round(time.time() - t0, 1)))
        print(f'[{ep:03d}] train loss {tr[0]:.4f} acc {tr[1]:.3f} | val loss {va[0]:.4f} '
              f'acc {va[1]:.3f} auc {va[2]} | {history[-1]["sec"]}s')
        if va[0] < best:
            best, bad = va[0], 0
            save_checkpoint(out_dir / 'best.pt', model, extra=dict(epoch=ep, args=vars(a)))
        else:
            bad += 1
            if bad >= Config.PATIENCE:
                print('early stopping')
                break

    (out_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (out_dir / 'train_config.json').write_text(json.dumps(
        dict(args=vars(a), seed=a.seed, n_train=len(ytr), n_pos=n_pos, n_neg=n_neg,
             img_size=Config.IMG_SIZE), indent=2), encoding='utf-8')
    print(f'best val loss {best:.4f} -> {out_dir / "best.pt"}')


if __name__ == '__main__':
    main()
