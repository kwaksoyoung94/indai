# rule_ssim / 5-fold 교차검증 (임계값 0.5)

주 집합: 영상 46개, 클립 1324개 (영상 단위 분할, 영상은 정확히 한 번 test). 통제된 불꽃 2개(164클립)는 제외하고 따로 보고: 클립 오경보율 0.805

| 지표 | 값 | 95% CI(영상 부트스트랩) |
|---|---|---|
| precision | 0.852 | [0.663, 0.936] |
| recall | 0.573 | [0.351, 0.759] |
| f1 | 0.685 | - |
| fpr | 0.502 | [0.311, 0.693] |
| auc | 0.534 | - |

## fold별
| fold | 클립 | precision | recall | FPR | AUC |
|---|---|---|---|---|---|
| 0 | 273 | 0.865 | 0.769 | 0.456 | 0.677 |
| 1 | 276 | 0.858 | 0.398 | 0.500 | 0.503 |
| 2 | 291 | 0.692 | 0.220 | 0.522 | 0.200 |
| 3 | 159 | 0.711 | 0.978 | 0.561 | 0.823 |
| 4 | 325 | 0.970 | 0.733 | 0.389 | 0.670 |

## 카테고리별 (화재=재현율, 비화재=오탐률)
| 카테고리 | 구분 | 영상 | 클립 | 재현율/오탐률 |
|---|---|---|---|---|
| beacon (non-fire) | non-fire | 1 | 15 | 1.000 |
| building (fire) | fire | 4 | 402 | 0.550 |
| factory_cctv (fire) | fire | 2 | 75 | 0.120 |
| factory_construction (non-fire) | non-fire | 1 | 11 | 0.000 |
| fireplace (non-fire) | non-fire | 1 | 150 | 0.813 |
| firesense (fire) | fire | 11 | 466 | 0.545 |
| firesense (non-fire) | non-fire | 9 | 70 | 0.871 |
| floodlight (non-fire) | non-fire | 1 | 4 | 1.000 |
| forklift_beacon (non-fire) | non-fire | 2 | 7 | 0.714 |
| forklift_linebeam (non-fire) | non-fire | 1 | 2 | 0.000 |
| gas_torch (non-fire) | non-fire | 1 | 14 | 0.714 |
| glass_reflection (non-fire) | non-fire | 2 | 34 | 0.176 |
| strobe (non-fire) | non-fire | 4 | 16 | 0.375 |
| stunt_flame (fire) | fire | 1 | 5 | 1.000 |
| sunlight (non-fire) | non-fire | 1 | 18 | 0.167 |
| sunlight_led (non-fire) | non-fire | 1 | 18 | 0.111 |
| vehicle (fire) | fire | 2 | 157 | 0.917 |
| welding (non-fire) | non-fire | 3 | 24 | 0.333 |