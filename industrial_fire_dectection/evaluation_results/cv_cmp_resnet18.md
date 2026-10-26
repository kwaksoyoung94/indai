# cmp_resnet18 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.896

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.976 | [0.921, 0.996] |
| recall | 0.577 | [0.370, 0.741] |
| f1 | 0.725 | - |
| fpr | 0.073 | [0.015, 0.148] |
| auc | 0.858 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.871 | 0.125 | 0.070 | 0.829 |
| 1 | 276 | 0.988 | 0.656 | 0.062 | 0.855 |
| 2 | 291 | 1.000 | 0.669 | 0.000 | 0.919 |
| 3 | 159 | 0.880 | 0.710 | 0.136 | 0.831 |
| 4 | 325 | 0.995 | 0.720 | 0.056 | 0.927 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 0.267 |
| building (fire) | fire | 4 | 402 | 0.614 |
| factory_cctv (fire) | fire | 2 | 75 | 0.160 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 0.980 |
| firesense (fire) | fire | 11 | 466 | 0.637 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.114 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.000 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.000 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.125 |
| stunt_flame (fire) | fire | 1 | 5 | 0.800 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.111 |
| vehicle (fire) | fire | 2 | 157 | 0.497 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |