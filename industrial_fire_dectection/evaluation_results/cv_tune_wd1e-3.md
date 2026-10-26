# tune_wd1e-3 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.951

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.947 | [0.820, 0.993] |
| recall | 0.498 | [0.245, 0.736] |
| f1 | 0.652 | - |
| fpr | 0.142 | [0.026, 0.300] |
| auc | 0.804 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.926 | 0.926 | 0.281 | 0.780 |
| 1 | 276 | 0.994 | 0.717 | 0.031 | 0.884 |
| 2 | 291 | 1.000 | 0.004 | 0.000 | 0.910 |
| 3 | 159 | 0.859 | 0.720 | 0.167 | 0.857 |
| 4 | 325 | 0.973 | 0.349 | 0.167 | 0.690 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.806 |
| factory_cctv (fire) | fire | 2 | 75 | 0.240 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.371 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.114 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.000 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.250 |
| stunt_flame (fire) | fire | 1 | 5 | 1.000 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.167 |
| vehicle (fire) | fire | 2 | 157 | 0.191 |
| welding (non-fire) | non-fire | 3 | 24 | 0.042 |