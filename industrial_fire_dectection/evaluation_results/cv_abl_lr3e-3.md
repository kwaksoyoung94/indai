# abl_lr3e-3 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.909

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.956 | [0.822, 0.998] |
| recall | 0.409 | [0.186, 0.630] |
| f1 | 0.573 | - |
| fpr | 0.096 | [0.005, 0.246] |
| auc | 0.822 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.921 | 0.810 | 0.263 | 0.814 |
| 1 | 276 | 0.994 | 0.676 | 0.031 | 0.876 |
| 2 | 291 | 0.974 | 0.151 | 0.022 | 0.906 |
| 3 | 159 | 0.600 | 0.065 | 0.061 | 0.692 |
| 4 | 325 | 1.000 | 0.225 | 0.000 | 0.836 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.771 |
| factory_cctv (fire) | fire | 2 | 75 | 0.213 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 0.993 |
| firesense (fire) | fire | 11 | 466 | 0.236 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.057 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.143 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.000 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.062 |
| stunt_flame (fire) | fire | 1 | 5 | 0.200 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.000 |
| vehicle (fire) | fire | 2 | 157 | 0.096 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |