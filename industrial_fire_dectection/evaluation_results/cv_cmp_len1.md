# cmp_len1 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.951

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.959 | [0.882, 0.994] |
| recall | 0.578 | [0.397, 0.737] |
| f1 | 0.722 | - |
| fpr | 0.123 | [0.024, 0.258] |
| auc | 0.819 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.924 | 0.736 | 0.228 | 0.798 |
| 1 | 276 | 0.994 | 0.713 | 0.031 | 0.863 |
| 2 | 291 | 1.000 | 0.698 | 0.000 | 0.920 |
| 3 | 159 | 0.787 | 0.398 | 0.152 | 0.694 |
| 4 | 325 | 0.970 | 0.319 | 0.167 | 0.660 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 0.867 |
| building (fire) | fire | 4 | 402 | 0.736 |
| factory_cctv (fire) | fire | 2 | 75 | 0.253 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.642 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.100 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.143 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.429 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.250 |
| stunt_flame (fire) | fire | 1 | 5 | 0.600 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.111 |
| vehicle (fire) | fire | 2 | 157 | 0.140 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |