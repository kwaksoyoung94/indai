# abl_mean / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.915

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.942 | [0.834, 0.994] |
| recall | 0.534 | [0.317, 0.736] |
| f1 | 0.682 | - |
| fpr | 0.164 | [0.024, 0.353] |
| auc | 0.797 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.918 | 0.778 | 0.263 | 0.801 |
| 1 | 276 | 0.958 | 0.840 | 0.281 | 0.885 |
| 2 | 291 | 1.000 | 0.535 | 0.000 | 0.929 |
| 3 | 159 | 0.727 | 0.344 | 0.182 | 0.682 |
| 4 | 325 | 1.000 | 0.176 | 0.000 | 0.636 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.764 |
| factory_cctv (fire) | fire | 2 | 75 | 0.533 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.496 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.257 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.000 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.000 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.125 |
| stunt_flame (fire) | fire | 1 | 5 | 0.600 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.056 |
| vehicle (fire) | fire | 2 | 157 | 0.057 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |