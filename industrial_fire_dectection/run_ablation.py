"""교수님 면담 지적 대응 보강 실험: 시간 모듈(LSTM/GRU/Transformer/평균 풀링)과 하이퍼파라미터 민감도.
각 변형을 5-fold 로 학습 -> fold별 평가 -> cv_report -> system_eval(conf 0.4). 이미 끝난 단계는 건너뛴다.

    python run_ablation.py            # 전부
    python run_ablation.py --only gru # 일부
모든 변형은 시드 42, 기존 대표 설정(은닉 64, lr 1e-3, 시퀀스 30)에서 한 가지씩만 바꾼다.
"""
import argparse
import subprocess
import sys

from config import Config

VARIANTS = {
    'abl_mean': ['--temporal', 'mean'],
    'abl_gru': ['--temporal', 'gru'],
    'abl_transformer': ['--temporal', 'transformer'],
    'abl_lr3e-4': ['--lr', '3e-4'],
    'abl_lr3e-3': ['--lr', '3e-3'],
    'abl_h32': ['--hidden', '32'],
    'abl_h128': ['--hidden', '128'],
    'tune_do0.3': ['--dropout', '0.3'],
    'tune_wd1e-3': ['--wd', '1e-3'],
    'tune_lr3e-4_h128': ['--lr', '3e-4', '--hidden', '128'],
}
K = 5


def run(cmd):
    print('>>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    py = sys.executable
    for name, extra in VARIANTS.items():
        if a.only and name.split('_', 1)[1] not in a.only and name not in a.only:
            continue
        for f in range(K):
            if not (Config.WEIGHTS_DIR / f'{name}_fold{f}' / 'best.pt').exists():
                run([py, 'train.py', '--name', name, '--fold', str(f), *extra])
            if not (Config.RESULTS_DIR / f'{name}_fold{f}_test' / 'predictions.csv').exists():
                run([py, 'evaluate.py', '--name', name, '--fold', str(f)])
        if not (Config.RESULTS_DIR / f'cv_{name}.json').exists():
            run([py, 'cv_report.py', '--name', name, '--k', str(K)])
        if not (Config.RESULTS_DIR / f'system_eval_{name}_conf0.4.json').exists():
            run([py, 'system_eval.py', '--name', name, '--conf', '0.4'])
        print(f'=== {name} 완료 ===', flush=True)


if __name__ == '__main__':
    main()
