"""교수님 지적 대응 비교 실험(4.2.1): 시간 정보 없음(프레임 1장), 표준 백본(ResNet18), 3D CNN(R3D-18).
대표 설정(시드 42, lr 1e-3, 배치 16, 드롭아웃 0.5)은 그대로 두고 모델 구성만 바꿔 5-fold 학습 -> fold별 평가 ->
cv_report -> system_eval(conf 0.4). 이미 끝난 단계는 건너뛴다.

    python run_compare.py              # 전부
    python run_compare.py --only len1  # 일부

cmp_len1 은 지속성 필터의 3초(30프레임) 조건은 그대로 두고 분류기에는 마지막 1프레임만 넣는다(--input-len 1).
"""
import argparse
import subprocess
import sys

from config import Config

# 이름: (학습 인자, evaluate.py 인자, system_eval.py 인자)
VARIANTS = {
    'cmp_len1': (['--seq-len', '1'], ['--seq-len', '1'], ['--input-len', '1']),
    'cmp_resnet18': (['--backbone', 'resnet18', '--pretrained'], [], []),
    'cmp_r3d18': (['--arch', 'r3d_18', '--pretrained'], [], []),
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
    for name, (tr, ev, se) in VARIANTS.items():
        if a.only and name.split('_', 1)[1] not in a.only and name not in a.only:
            continue
        for f in range(K):
            if not (Config.WEIGHTS_DIR / f'{name}_fold{f}' / 'best.pt').exists():
                run([py, 'train.py', '--name', name, '--fold', str(f), *tr])
            if not (Config.RESULTS_DIR / f'{name}_fold{f}_test' / 'predictions.csv').exists():
                run([py, 'evaluate.py', '--name', name, '--fold', str(f), *ev])
        if not (Config.RESULTS_DIR / f'cv_{name}.json').exists():
            run([py, 'cv_report.py', '--name', name, '--k', str(K)])
        if not (Config.RESULTS_DIR / f'system_eval_{name}_conf0.4.json').exists():
            run([py, 'system_eval.py', '--name', name, '--conf', '0.4', *se])
        print(f'=== {name} 완료 ===', flush=True)


if __name__ == '__main__':
    main()
