# 시스템 단위 평가: cnn_lstm, YOLO conf 0.15, Stage2 임계값 0.75

영상 61개: 화재 20, 비화재 주 집합 39 (통제된 불꽃 2개 분리)

| 방법 | 화재 검출(재현율) | **주 오경보율** (n=39) | 95% CI | 통제된 불꽃 오경보 (n=2) | 전체 비화재 오경보율 (n=41) | 화재 경보까지 중앙값(초) |
|---|---|---|---|---|---|---|
| yolo_only | 1.000 (20/20) | **0.846** (33/39) | [0.72, 0.95] | 2/2 | 0.854 (35/41) | 0.2 |
| persistence | 1.000 (20/20) | **0.667** (26/39) | [0.51, 0.82] | 2/2 | 0.683 (28/41) | 3.0 |
| stage2_c1 | 0.950 (19/20) | **0.282** (11/39) | [0.13, 0.44] | 2/2 | 0.317 (13/41) | 3.0 |
| stage2_c2 | 0.950 (19/20) | **0.231** (9/39) | [0.10, 0.36] | 2/2 | 0.268 (11/41) | 4.5 |
| stage2_c3 | 0.950 (19/20) | **0.128** (5/39) | [0.03, 0.23] | 2/2 | 0.171 (7/41) | 5.0 |

## 카테고리별 오경보 영상 수 / 영상 수 (비화재)
| 카테고리 | yolo_only | persistence | stage2_c1 | stage2_c2 | stage2_c3 |
|---|---|---|---|---|---|
| interference/agv | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| interference/beacon | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| interference/fireplace (통제된 불꽃) | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| interference/floodlight | 2/2 | 1/2 | 0/2 | 0/2 | 0/2 |
| interference/forklift_beacon | 2/3 | 2/3 | 2/3 | 2/3 | 1/3 |
| interference/forklift_linebeam | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| interference/gas_torch (통제된 불꽃) | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| interference/glass_reflection | 3/3 | 2/3 | 1/3 | 1/3 | 0/3 |
| interference/strobe | 4/4 | 4/4 | 3/4 | 2/4 | 2/4 |
| interference/sunlight | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 |
| interference/sunlight_led | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| interference/welding | 3/3 | 3/3 | 0/3 | 0/3 | 0/3 |
| normal/factory_construction | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 |
| normal/firesense | 12/16 | 9/16 | 2/16 | 1/16 | 1/16 |

## 화재 영상 검출 (카테고리별 검출 영상 수 / 영상 수)
| 카테고리 | yolo_only | persistence | stage2_c1 | stage2_c2 | stage2_c3 |
|---|---|---|---|---|---|
| fire/building | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| fire/factory_cctv | 2/2 | 2/2 | 1/2 | 1/2 | 1/2 |
| fire/firesense | 11/11 | 11/11 | 11/11 | 11/11 | 11/11 |
| fire/stunt_flame | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| fire/vehicle | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |