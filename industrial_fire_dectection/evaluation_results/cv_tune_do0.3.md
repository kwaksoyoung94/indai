# tune_do0.3 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.951

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.956 | [0.870, 0.995] |
| recall | 0.605 | [0.438, 0.772] |
| f1 | 0.741 | - |
| fpr | 0.142 | [0.022, 0.300] |
| auc | 0.820 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.919 | 0.787 | 0.263 | 0.749 |
| 1 | 276 | 0.995 | 0.783 | 0.031 | 0.908 |
| 2 | 291 | 0.985 | 0.547 | 0.043 | 0.967 |
| 3 | 159 | 0.800 | 0.516 | 0.182 | 0.688 |
| 4 | 325 | 0.992 | 0.410 | 0.056 | 0.701 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.799 |
| factory_cctv (fire) | fire | 2 | 75 | 0.373 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.575 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.114 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.286 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.125 |
| stunt_flame (fire) | fire | 1 | 5 | 1.000 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.222 |
| vehicle (fire) | fire | 2 | 157 | 0.299 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |