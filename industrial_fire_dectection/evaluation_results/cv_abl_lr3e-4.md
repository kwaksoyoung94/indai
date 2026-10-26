# abl_lr3e-4 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.951

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.960 | [0.873, 0.991] |
| recall | 0.526 | [0.309, 0.727] |
| f1 | 0.680 | - |
| fpr | 0.110 | [0.028, 0.233] |
| auc | 0.807 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | - | 0.000 | 0.000 | 0.814 |
| 1 | 276 | 0.980 | 0.811 | 0.125 | 0.896 |
| 2 | 291 | 0.977 | 0.531 | 0.065 | 0.866 |
| 3 | 159 | 0.775 | 0.591 | 0.242 | 0.716 |
| 4 | 325 | 0.995 | 0.645 | 0.056 | 0.815 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 0.000 |
| building (fire) | fire | 4 | 402 | 0.575 |
| factory_cctv (fire) | fire | 2 | 75 | 0.480 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.532 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.214 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.429 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.125 |
| stunt_flame (fire) | fire | 1 | 5 | 1.000 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.222 |
| vehicle (fire) | fire | 2 | 157 | 0.389 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |