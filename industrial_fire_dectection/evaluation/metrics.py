import numpy as np
from sklearn.metrics import roc_auc_score


def _div(a, b):
    return float(a / b) if b else None


def binary_metrics(y, prob, thr):
    """y: 0/1 배열, prob: 화재 확률. 정의되지 않는 지표(분모 0)는 None으로 반환한다."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = _div(2 * p * r, p + r) if p is not None and r is not None else None
    auc = float(roc_auc_score(y, prob)) if len(set(y.tolist())) == 2 else None
    return dict(n=len(y), n_pos=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
                precision=p, recall=r, f1=f1, fpr=_div(fp, fp + tn),
                accuracy=_div(tp + tn, len(y)), auc=auc, threshold=float(thr))


def video_level(df, agg='mean'):
    """클립 확률을 영상 단위로 집계. agg: mean | max"""
    g = df.groupby('video_id').agg(y=('y', 'first'), prob=('prob', agg)).reset_index()
    return g


def bootstrap_ci(df, thr, keys=('fpr', 'recall', 'precision'), n_boot=1000, seed=0):
    """영상 단위 부트스트랩 95% 신뢰구간(같은 영상의 클립은 독립이 아니므로 영상을 리샘플)."""
    rng = np.random.default_rng(seed)
    vids = df.video_id.unique()
    groups = {v: g for v, g in df.groupby('video_id')}
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        s = df.__class__(np.concatenate([groups[v].values for v in pick]), columns=df.columns)
        m = binary_metrics(s.y.astype(int), s.prob.astype(float), thr)
        for k in keys:
            if m[k] is not None:
                vals[k].append(m[k])
    return {k: (None if len(v) < 20 else [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
            for k, v in vals.items()}
