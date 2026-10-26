"""프레임 간 SSIM/MSE/밝기 시계열과 주기성 특징 (규칙 기반 시간 분석).

PPT/논문의 '프레임 간 변화율, SSIM, MSE로 주기적 광원을 판별한다'는 설명을 실제로 구현한 모듈.
(1) 규칙 기반 비교 모델(baselines.py)의 입력, (2) 논문 그림용 SSIM 시계열 분석에 사용한다.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity

FEATURE_NAMES = ['ssim_mean', 'ssim_std', 'mse_mean', 'mse_std', 'bright_mean', 'bright_std',
                 'ssim_peak_ratio', 'ssim_spec_entropy', 'bright_peak_ratio', 'bright_spec_entropy',
                 'ssim_roughness']


def _gray(clip):
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in clip])


def ssim_series(clip):
    g = _gray(clip)
    return np.array([structural_similarity(g[t - 1], g[t], data_range=255) for t in range(1, len(g))])


def mse_series(clip):
    g = _gray(clip).astype(np.float32)
    return ((g[1:] - g[:-1]) ** 2).mean(axis=(1, 2))


def brightness_series(clip):
    return _gray(clip).mean(axis=(1, 2))


def periodicity(series):
    """(peak_ratio, 정규화 스펙트럼 엔트로피). 값이 일정한 주파수에 몰릴수록 peak_ratio가 크고
    엔트로피가 작다. 평탄한 신호는 (0, 1)로 처리한다."""
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    if np.allclose(x, 0, atol=1e-8) or len(x) < 4:
        return 0.0, 1.0
    p = np.abs(np.fft.rfft(x))[1:] ** 2
    if p.sum() <= 0:
        return 0.0, 1.0
    q = p / p.sum()
    ent = -(q * np.log(q + 1e-12)).sum() / np.log(len(q))
    return float(q.max()), float(ent)


def clip_features(clip):
    s, m, b = ssim_series(clip), mse_series(clip), brightness_series(clip)
    s_pk, s_en = periodicity(s)
    b_pk, b_en = periodicity(b)
    rough = float(np.abs(np.diff(s)).mean()) if len(s) > 1 else 0.0
    return np.array([s.mean(), s.std(), m.mean(), m.std(), b.mean(), b.std(),
                     s_pk, s_en, b_pk, b_en, rough], dtype=np.float64)
