# 시스템 단위 평가: tune_lr3e-4_h128, YOLO conf 0.4, Stage2 임계값 0.75

영상 61개: 화재 20, 비화재 주 집합 39 (통제된 불꽃 2개 분리)

| 방법 | 화재 검출(재현율) | **주 오경보율** (n=39) | 95% CI | 통제된 불꽃 오경보 (n=2) | 전체 비화재 오경보율 (n=41) | 화재 경보까지 중앙값(초) |
|---|---|---|---|---|---|---|
| yolo_only | 1.000 (20/20) | **0.538** (21/39) | [0.38, 0.69] | 2/2 | 0.561 (23/41) | 0.2 |
| persistence | 1.000 (20/20) | **0.308** (12/39) | [0.18, 0.46] | 2/2 | 0.341 (14/41) | 2.9 |
| stage2_c1 | 0.650 (13/20) | **0.077** (3/39) | [0.00, 0.15] | 1/2 | 0.098 (4/41) | 6.0 |
| stage2_c2 | 0.500 (10/20) | **0.077** (3/39) | [0.00, 0.18] | 1/2 | 0.098 (4/41) | 5.0 |
| stage2_c3 | 0.500 (10/20) | **0.026** (1/39) | [0.00, 0.08] | 1/2 | 0.049 (2/41) | 5.5 |

## 카테고리별 오경보 영상 수 / 영상 수 (비화재)
| 카테고리 | yolo_only | persistence | stage2_c1 | stage2_c2 | stage2_c3 |
|---|---|---|---|---|---|
| interference/agv | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| interference/beacon | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| interference/fireplace (통제된 불꽃) | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| interference/floodlight | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| interference/forklift_beacon | 2/3 | 1/3 | 0/3 | 0/3 | 0/3 |
| interference/forklift_linebeam | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| interference/gas_torch (통제된 불꽃) | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| interference/glass_reflection | 2/3 | 2/3 | 0/3 | 0/3 | 0/3 |
| interference/strobe | 3/4 | 1/4 | 0/4 | 0/4 | 0/4 |
| interference/sunlight | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| interference/sunlight_led | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| interference/welding | 2/3 | 1/3 | 0/3 | 0/3 | 0/3 |
| normal/factory_construction | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| normal/firesense | 8/16 | 3/16 | 0/16 | 0/16 | 0/16 |

## 화재 영상 검출 (카테고리별 검출 영상 수 / 영상 수)
| 카테고리 | yolo_only | persistence | stage2_c1 | stage2_c2 | stage2_c3 |
|---|---|---|---|---|---|
| fire/building | 4/4 | 4/4 | 3/4 | 2/4 | 2/4 |
| fire/factory_cctv | 2/2 | 2/2 | 1/2 | 1/2 | 1/2 |
| fire/firesense | 11/11 | 11/11 | 8/11 | 7/11 | 7/11 |
| fire/stunt_flame | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 |
| fire/vehicle | 2/2 | 2/2 | 0/2 | 0/2 | 0/2 |