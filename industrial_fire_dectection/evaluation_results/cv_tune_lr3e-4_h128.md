# tune_lr3e-4_h128 / 5-fold 교차검증 (임계값 0.75)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.915

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.945 | [0.833, 0.987] |
| recall | 0.511 | [0.271, 0.726] |
| f1 | 0.664 | - |
| fpr | 0.151 | [0.044, 0.276] |
| auc | 0.788 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.928 | 0.713 | 0.211 | 0.829 |
| 1 | 276 | 0.974 | 0.758 | 0.156 | 0.838 |
| 2 | 291 | 0.994 | 0.678 | 0.022 | 0.930 |
| 3 | 159 | 0.797 | 0.634 | 0.227 | 0.739 |
| 4 | 325 | 1.000 | 0.003 | 0.000 | 0.611 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 0.800 |
| building (fire) | fire | 4 | 402 | 0.652 |
| factory_cctv (fire) | fire | 2 | 75 | 0.360 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.182 |
| fireplace (non-fire) | non-fire | 1 | 150 | 1.000 |
| firesense (fire) | fire | 11 | 466 | 0.590 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.214 |
| floodlight (non-fire) | non-fire | 1 | 4 | 0.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.143 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.000 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.000 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.062 |
| stunt_flame (fire) | fire | 1 | 5 | 0.200 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.000 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.111 |
| vehicle (fire) | fire | 2 | 157 | 0.000 |
| welding (non-fire) | non-fire | 3 | 24 | 0.000 |