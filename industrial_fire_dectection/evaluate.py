"""학습된 CNN-LSTM 을 지정 split(기본 test)에서 평가하고 결과를 evaluation_results/<run>/ 에 저장.

저장물: predictions.csv, metrics.json(클립/영상 단위, 부트스트랩 CI, 임계값 스윕),
        confusion_matrix.png, roc.png

    python evaluate.py --name cnn_lstm --split test
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from config import Config
from dataset.data_loader import get_loader
from evaluation.metrics import binary_metrics, bootstrap_ci, video_level
from models.cnn_lstm import load_checkpoint


@torch.inference_mode()
def predict(model, loader, device):
    probs = []
    for x, _, _ in loader:
        probs.append(torch.sigmoid(model(x.to(device)).float()).cpu())
    return torch.cat(probs).numpy()


def write_report(out_dir, df, thr, title=''):
    """df: video_id, y, prob (+ clip_path). 지표/그림 저장 후 metrics dict 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'predictions.csv', index=False)
    clip = binary_metrics(df.y, df.prob, thr)
    vid_mean = binary_metrics(*[video_level(df, 'mean')[c] for c in ('y', 'prob')], thr)
    vid_max = binary_metrics(*[video_level(df, 'max')[c] for c in ('y', 'prob')], thr)
    sweep = {f'{t:.2f}': {k: binary_metrics(df.y, df.prob, t)[k] for k in ('precision', 'recall', 'fpr')}
             for t in (0.25, 0.5, 0.75, 0.9)}
    metrics = dict(title=title, n_videos=int(df.video_id.nunique()), clip_level=clip,
                   video_level_mean=vid_mean, video_level_max=vid_max,
                   clip_level_ci95_video_bootstrap=bootstrap_ci(df, thr), threshold_sweep=sweep)
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')

    cm = np.array([[clip['tn'], clip['fp']], [clip['fn'], clip['tp']]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha='center', va='center', color='white' if v > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['Non-fire', 'Fire'])
    ax.set_yticks([0, 1], ['Non-fire', 'Fire'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion matrix (thr={thr})')
    fig.tight_layout()
    fig.savefig(out_dir / 'confusion_matrix.png', dpi=200)
    plt.close(fig)

    if clip['auc'] is not None:
        fpr, tpr, _ = roc_curve(df.y, df.prob)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.plot(fpr, tpr, label=f'AUC={clip["auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k:')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'roc.png', dpi=200)
        plt.close(fig)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cnn_lstm')
    ap.add_argument('--split', default='test')
    ap.add_argument('--threshold', type=float, default=Config.FIRE_THRESHOLD)
    ap.add_argument('--seq-len', type=int, default=Config.SEQUENCE_LENGTH)
    ap.add_argument('--fold', type=int, default=None)
    a = ap.parse_args()
    if a.fold is not None:
        a.name = f'{a.name}_fold{a.fold}'

    device = Config.DEVICE
    model, _ = load_checkpoint(Config.WEIGHTS_DIR / a.name / 'best.pt', device)
    loader = get_loader(a.split, seq_len=a.seq_len, fold=a.fold)
    prob = predict(model, loader, device)
    d = loader.dataset.df
    df = pd.DataFrame(dict(clip_path=d.clip_path, video_id=d.video_id, label=d.label, y=d.y, prob=prob))
    m = write_report(Config.RESULTS_DIR / f'{a.name}_{a.split}', df, a.threshold, title=a.name)
    c = m['clip_level']
    print(f'[{a.name}/{a.split}] clips={c["n"]} videos={m["n_videos"]}  '
          f'P={c["precision"]} R={c["recall"]} F1={c["f1"]} FPR={c["fpr"]} AUC={c["auc"]}')


if __name__ == '__main__':
    main()
