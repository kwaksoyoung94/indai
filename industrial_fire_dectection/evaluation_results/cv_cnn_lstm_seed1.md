# cnn_lstm_seed1 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.951

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.949 | [0.864, 0.991] |
| recall | 0.680 | [0.526, 0.837] |
| f1 | 0.792 | - |
| fpr | 0.183 | [0.039, 0.359] |
| auc | 0.836 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.925 | 0.852 | 0.263 | 0.829 |
| 1 | 276 | 0.990 | 0.779 | 0.062 | 0.911 |
| 2 | 291 | 0.989 | 0.722 | 0.043 | 0.833 |
| 3 | 159 | 0.673 | 0.398 | 0.273 | 0.700 |
| 4 | 325 | 0.982 | 0.531 | 0.167 | 0.769 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.853 |
| factory_cctv (fire) | fire | 2 | 75 | 0.373 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.710 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.171 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.286 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.312 |
| stunt_flame (fire) | fire | 1 | 5 | 0.600 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.333 |
| vehicle (fire) | fire | 2 | 157 | 0.293 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |