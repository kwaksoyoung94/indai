"""Stage 1 YOLOv8 화재/연기 탐지기 학습 (D-Fire 등 YOLO 포맷 데이터셋).

기대하는 폴더 (D-Fire Kaggle/OneDrive 배포본의 train/val/test 분할):
    dataset/external/dfire/{train,val,test}/{images,labels}/
클래스: 0=smoke, 1=fire (D-Fire 규약). 배포본의 순서가 다르면 --names 로 지정.

    python train_yolo.py --data-dir dataset/external/dfire --epochs 50
결과 가중치는 models/weights/yolov8n_fire.pt 로 복사되어 Stage 1에서 바로 사용된다.
"""
import argparse
import shutil
from pathlib import Path

import yaml

from config import Config


def make_train_val_lists(root, val_frac, block, seed):
    """val 폴더가 없는 배포본(D-Fire는 train/test만 제공)에서 train의 일부를 val로 분리한다.

    같은 영상에서 뽑은 인접 프레임이 train/val에 갈라지면 val 성능이 부풀려지므로, 이미지를 무작위로
    뽑지 않고 접두사별로 번호순 정렬한 뒤 연속 구간(block장) 단위로 val에 배정한다.
    파일 번호가 영상 순서를 따른다는 가정에 의존하므로 완전한 누수 차단은 아니다(논문 한계로 기술).
    """
    import random
    import re
    imgs = sorted((root / 'train' / 'images').glob('*.jpg'),
                  key=lambda p: (re.sub(r'\d+', '', p.stem), int(re.findall(r'\d+', p.stem)[-1])
                                 if re.findall(r'\d+', p.stem) else 0, p.stem))
    blocks = [imgs[i:i + block] for i in range(0, len(imgs), block)]
    rng = random.Random(seed)
    val_ids = set(rng.sample(range(len(blocks)), max(1, round(len(blocks) * val_frac))))
    train = [p for i, b in enumerate(blocks) if i not in val_ids for p in b]
    val = [p for i, b in enumerate(blocks) if i in val_ids for p in b]
    for name, items in (('train_split.txt', train), ('val_split.txt', val)):
        (root / name).write_text('\n'.join(p.as_posix() for p in items), encoding='utf-8')
    print(f'train {len(train)}장 / val {len(val)}장 (연속 {block}장 구간 단위 분리)')
    return dict(train='train_split.txt', val='val_split.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=2, help='Windows에서 8개는 페이지파일 부족(오류 1455)으로 교착됨')
    ap.add_argument('--run-name', default='train')
    ap.add_argument('--fraction', type=float, default=1.0, help='학습 데이터 사용 비율(스모크 테스트용)')
    ap.add_argument('--no-install', action='store_true', help='yolov8n_fire.pt 로 복사하지 않음')
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--val-block', type=int, default=50)
    ap.add_argument('--data-dir', default=str(Config.DATA_DIR / 'external' / 'dfire'))
    ap.add_argument('--model', default='yolov8n.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--names', nargs='+', default=['smoke', 'fire'])
    ap.add_argument('--seed', type=int, default=Config.SEED)
    a = ap.parse_args()

    root = Path(a.data_dir).resolve()
    if not (root / 'train' / 'images').is_dir():
        raise SystemExit(f'{root / "train" / "images"} 가 없습니다. D-Fire를 이 위치에 풀어 주세요.')
    cfg = {'path': str(root), 'names': dict(enumerate(a.names))}
    if (root / 'val' / 'images').is_dir():
        cfg.update(train='train/images', val='val/images')
    else:
        cfg.update(make_train_val_lists(root, a.val_frac, a.val_block, a.seed))
    if (root / 'test' / 'images').is_dir():
        cfg['test'] = 'test/images'
    yaml_path = root / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')

    from ultralytics import YOLO
    model = YOLO(a.model)
    run_dir = Config.RESULTS_DIR / 'yolo'
    model.train(data=str(yaml_path), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed,
                device=0 if Config.DEVICE == 'cuda' else 'cpu', fraction=a.fraction, workers=a.workers,
                project=str(run_dir), name=a.run_name, exist_ok=True)
    best = run_dir / a.run_name / 'weights' / 'best.pt'
    if not a.no_install:
        Config.YOLO_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, Config.YOLO_MODEL_PATH)
        print(f'가중치 저장: {Config.YOLO_MODEL_PATH}')
    if 'test' in cfg:
        m = model.val(data=str(yaml_path), split='test', project=str(run_dir), name='test', exist_ok=True)
        print(f'test mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}')


if __name__ == '__main__':
    main()
