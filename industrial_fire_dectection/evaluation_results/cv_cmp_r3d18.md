# cmp_r3d18 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.939

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.927 | [0.744, 0.983] |
| recall | 0.471 | [0.183, 0.732] |
| f1 | 0.624 | - |
| fpr | 0.187 | [0.051, 0.342] |
| auc | 0.630 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.845 | 0.861 | 0.596 | 0.624 |
| 1 | 276 | 0.984 | 0.762 | 0.094 | 0.893 |
| 2 | 291 | - | 0.000 | 0.000 | 0.245 |
| 3 | 159 | 0.921 | 0.376 | 0.045 | 0.770 |
| 4 | 325 | 0.991 | 0.368 | 0.056 | 0.720 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 0.867 |
| building (fire) | fire | 4 | 402 | 0.915 |
| factory_cctv (fire) | fire | 2 | 75 | 0.280 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.249 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.186 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.000 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.286 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.294 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.125 |
| stunt_flame (fire) | fire | 1 | 5 | 0.000 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.111 |
| vehicle (fire) | fire | 2 | 157 | 0.096 |
| welding (non-fire) | non-fire | 3 | 24 | 0.042 |