"""비교 모델: ROI 클립에서 SSIM/MSE/주기성 특징을 뽑아 로지스틱 회귀로 판별하는 규칙 기반 시간 분석.

CNN-LSTM 과 동일한 train/test 클립·분할·지표로 평가해 '학습 기반 시공간 모델이 필요한가'를 검증한다.

    python baselines.py --name rule_ssim --split test
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import Config
from dataset.data_loader import clips_for
from evaluate import write_report
from models.features import FEATURE_NAMES, clip_features


def featurize(split, seq_len, fold=None):
    d = clips_for(split, fold)
    X = np.stack([clip_features(np.load(Config.ROI_DIR / p)[-seq_len:])
                  for p in tqdm(d.clip_path, desc=f'features[{split}]')])
    return d, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='rule_ssim')
    ap.add_argument('--split', default='test')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--seq-len', type=int, default=Config.SEQUENCE_LENGTH)
    ap.add_argument('--fold', type=int, default=None)
    a = ap.parse_args()
    if a.fold is not None:
        a.name = f'{a.name}_fold{a.fold}'

    dtr, Xtr = featurize('train', a.seq_len, a.fold)
    dte, Xte = featurize(a.split, a.seq_len, a.fold)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000))
    clf.fit(Xtr, dtr.y)
    coef = dict(zip(FEATURE_NAMES, clf[-1].coef_[0].round(3)))
    prob = clf.predict_proba(Xte)[:, 1]
    df = pd.DataFrame(dict(clip_path=dte.clip_path, video_id=dte.video_id, label=dte.label,
                           y=dte.y, prob=prob))
    m = write_report(Config.RESULTS_DIR / f'{a.name}_{a.split}', df, a.threshold, title=a.name)
    c = m['clip_level']
    print(f'[{a.name}/{a.split}] P={c["precision"]} R={c["recall"]} FPR={c["fpr"]} AUC={c["auc"]}')
    print('특징 계수:', coef)


if __name__ == '__main__':
    main()
