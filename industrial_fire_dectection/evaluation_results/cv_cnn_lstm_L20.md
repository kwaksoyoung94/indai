# cnn_lstm_L20 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.933

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.947 | [0.856, 0.992] |
| recall | 0.619 | [0.457, 0.765] |
| f1 | 0.749 | - |
| fpr | 0.174 | [0.034, 0.332] |
| auc | 0.799 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.920 | 0.690 | 0.228 | 0.803 |
| 1 | 276 | 0.995 | 0.746 | 0.031 | 0.891 |
| 2 | 291 | 0.983 | 0.727 | 0.065 | 0.791 |
| 3 | 159 | 0.750 | 0.613 | 0.288 | 0.648 |
| 4 | 325 | 0.983 | 0.384 | 0.111 | 0.827 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 0.867 |
| building (fire) | fire | 4 | 402 | 0.739 |
| factory_cctv (fire) | fire | 2 | 75 | 0.293 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.693 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.157 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.429 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.214 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.188 |
| stunt_flame (fire) | fire | 1 | 5 | 0.800 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.444 |
| vehicle (fire) | fire | 2 | 157 | 0.242 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |