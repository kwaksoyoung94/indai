# cnn_lstm_L10 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.951

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.945 | [0.844, 0.989] |
| recall | 0.632 | [0.473, 0.787] |
| f1 | 0.757 | - |
| fpr | 0.187 | [0.046, 0.377] |
| auc | 0.796 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.921 | 0.810 | 0.263 | 0.818 |
| 1 | 276 | 0.994 | 0.734 | 0.031 | 0.884 |
| 2 | 291 | 0.989 | 0.706 | 0.043 | 0.838 |
| 3 | 159 | 0.622 | 0.301 | 0.258 | 0.537 |
| 4 | 325 | 0.960 | 0.466 | 0.333 | 0.639 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.808 |
| factory_cctv (fire) | fire | 2 | 75 | 0.280 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.637 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.229 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.429 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.250 |
| stunt_flame (fire) | fire | 1 | 5 | 0.800 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.167 |
| vehicle (fire) | fire | 2 | 157 | 0.325 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |