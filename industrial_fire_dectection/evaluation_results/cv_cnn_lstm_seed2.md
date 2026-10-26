# cnn_lstm_seed2 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.945

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.943 | [0.843, 0.989] |
| recall | 0.589 | [0.424, 0.756] |
| f1 | 0.725 | - |
| fpr | 0.178 | [0.044, 0.345] |
| auc | 0.782 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.919 | 0.894 | 0.298 | 0.834 |
| 1 | 276 | 0.994 | 0.631 | 0.031 | 0.866 |
| 2 | 291 | 0.989 | 0.702 | 0.043 | 0.823 |
| 3 | 159 | 0.484 | 0.161 | 0.242 | 0.470 |
| 4 | 325 | 0.975 | 0.381 | 0.167 | 0.823 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.794 |
| factory_cctv (fire) | fire | 2 | 75 | 0.227 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 0.993 |
| firesense (fire) | fire | 11 | 466 | 0.590 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.214 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.286 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.059 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.250 |
| stunt_flame (fire) | fire | 1 | 5 | 0.200 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.000 |
| vehicle (fire) | fire | 2 | 157 | 0.248 |
| welding (non-fire) | non-fire | 3 | 24 | 0.042 |