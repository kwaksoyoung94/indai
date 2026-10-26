# cnn_lstm / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.951

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.960 | [0.887, 0.991] |
| recall | 0.700 | [0.560, 0.843] |
| f1 | 0.810 | - |
| fpr | 0.146 | [0.049, 0.300] |
| auc | 0.817 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.922 | 0.935 | 0.298 | 0.809 |
| 1 | 276 | 0.989 | 0.770 | 0.062 | 0.858 |
| 2 | 291 | 0.988 | 0.678 | 0.043 | 0.756 |
| 3 | 159 | 0.877 | 0.538 | 0.106 | 0.818 |
| 4 | 325 | 0.977 | 0.547 | 0.222 | 0.797 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.891 |
| factory_cctv (fire) | fire | 2 | 75 | 0.360 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.704 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.057 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.286 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.029 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.312 |
| stunt_flame (fire) | fire | 1 | 5 | 0.800 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.222 |
| vehicle (fire) | fire | 2 | 157 | 0.363 |
| welding (non-fire) | non-fire | 3 | 24 | 0.042 |