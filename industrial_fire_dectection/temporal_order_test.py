"""학습된 CNN-LSTM 이 프레임 순서(시간 정보)를 실제로 쓰는지 확인한다.

각 fold 의 test 클립에서 프레임 순서를 (a) 원래대로, (b) 무작위로 섞어서, (c) 거꾸로 넣어 같은 학습 모델의 예측을 비교한다.
순서를 바꿔도 성능이 같다면 모델은 시간 순서를 쓰지 않고 프레임 한 장의 모양만 본다는 뜻이다. 모델과 데이터는 그대로이고
입력 순서만 바꾸므로 재학습이 필요 없다. 주 집합(통제된 불꽃 제외)에서 클립 단위와 영상 단위(클립 확률 평균) AUC 를 낸다.

    python temporal_order_test.py --name cnn_lstm --k 5 --repeats 20
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from config import Config
from dataset import manifest
from dataset.data_loader import get_loader
from models.cnn_lstm import load_checkpoint


@torch.inference_mode()
def run(model, loader, device, mode, gen):
    probs = []
    for x, _, _ in loader:
        x = x.to(device)
        t = x.shape[1]
        if mode == 'shuffle':
            idx = torch.stack([torch.randperm(t, generator=gen) for _ in range(len(x))]).to(device)
            x = torch.gather(x, 1, idx.view(len(x), t, 1, 1, 1).expand_as(x))
        elif mode == 'reverse':
            x = x.flip(1)
        probs.append(torch.sigmoid(model(x).float()).cpu())
    return torch.cat(probs).numpy()


def aucs(df):
    v = df.groupby('video_id').agg(y=('y', 'first'), prob=('prob', 'mean'))
    return float(roc_auc_score(df.y, df.prob)), float(roc_auc_score(v.y, v.prob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cnn_lstm')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    device = Config.DEVICE
    cat = manifest.load().set_index('video_id').category
    gen = torch.Generator().manual_seed(a.seed)

    runs = {'original': [], 'reverse': [], **{f'shuffle{r}': [] for r in range(a.repeats)}}
    meta = []
    for f in range(a.k):
        model, _ = load_checkpoint(Config.WEIGHTS_DIR / f'{a.name}_fold{f}' / 'best.pt', device)
        model.eval()
        loader = get_loader('test', fold=f)
        d = loader.dataset.df
        meta.append(pd.DataFrame(dict(video_id=d.video_id.values, y=d.y.values)))
        runs['original'].append(run(model, loader, device, 'orig', gen))
        runs['reverse'].append(run(model, loader, device, 'reverse', gen))
        for r in range(a.repeats):
            runs[f'shuffle{r}'].append(run(model, loader, device, 'shuffle', gen))
    meta = pd.concat(meta, ignore_index=True)
    keep = ~meta.video_id.map(cat).isin(Config.CONTROLLED_FLAME_CATEGORIES).values
    out = {}
    for k, v in runs.items():
        df = meta.copy()
        df['prob'] = np.concatenate(v)
        df = df[keep]
        out[k] = dict(zip(('clip_auc', 'video_auc'), aucs(df)))
        out[k]['fpr_at_thr'] = float(((df.prob >= Config.FIRE_THRESHOLD) & (df.y == 0)).sum() / (df.y == 0).sum())
        out[k]['recall_at_thr'] = float(((df.prob >= Config.FIRE_THRESHOLD) & (df.y == 1)).sum() / (df.y == 1).sum())
    sh = [out[f'shuffle{r}'] for r in range(a.repeats)]
    summary = dict(name=a.name, repeats=a.repeats, n_clips=int(keep.sum()), n_videos=int(meta[keep].video_id.nunique()),
                   original=out['original'], reverse=out['reverse'],
                   shuffle_mean={m: float(np.mean([s[m] for s in sh])) for m in sh[0]},
                   shuffle_std={m: float(np.std([s[m] for s in sh])) for m in sh[0]})
    (Config.RESULTS_DIR / f'temporal_order_{a.name}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
