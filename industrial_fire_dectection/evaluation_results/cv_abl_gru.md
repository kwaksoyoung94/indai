# abl_gru / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.933

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.967 | [0.891, 0.995] |
| recall | 0.633 | [0.471, 0.774] |
| f1 | 0.765 | - |
| fpr | 0.110 | [0.022, 0.237] |
| auc | 0.817 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.925 | 0.796 | 0.246 | 0.898 |
| 1 | 276 | 0.989 | 0.738 | 0.062 | 0.837 |
| 2 | 291 | 0.995 | 0.784 | 0.022 | 0.839 |
| 3 | 159 | 0.829 | 0.366 | 0.106 | 0.613 |
| 4 | 325 | 1.000 | 0.394 | 0.000 | 0.864 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 0.933 |
| building (fire) | fire | 4 | 402 | 0.801 |
| factory_cctv (fire) | fire | 2 | 75 | 0.293 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.091 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.655 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.071 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.143 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.214 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.062 |
| stunt_flame (fire) | fire | 1 | 5 | 0.800 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.111 |
| vehicle (fire) | fire | 2 | 157 | 0.293 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |