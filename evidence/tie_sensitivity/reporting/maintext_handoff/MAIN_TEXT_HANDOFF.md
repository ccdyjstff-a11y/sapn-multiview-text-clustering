# 主文整合结果索引

这是事后敏感性分析，primary保留。全部拟合先封存，后统一评分与独立核验；不构成独立确认。

## 20项正均值计数

| 政策 | 控制类型 | 正 / 零 / 负均值 | 完整 / 计划 |
|---|---|---|---|
| primary_default | independent_CV | 19 / 0 / 1 | 20 / 20 |
| primary_default | policy_Full_lambda | 19 / 0 / 1 | 20 / 20 |
| sensitivity_min | independent_CV | 18 / 0 / 2 | 20 / 20 |
| sensitivity_min | policy_Full_lambda | 19 / 0 / 1 | 20 / 20 |
| sensitivity_max | independent_CV | 16 / 0 / 4 | 20 / 20 |
| sensitivity_max | policy_Full_lambda | 19 / 0 / 1 | 20 / 20 |

## 相对primary方向改变

零均值容差为1e-12个百分点。完整名单见 ACC_direction_changes_from_primary.csv。

| 语料 | Q | 控制 | 替代规则 | primary均值 | 替代均值 |
|---|---:|---|---|---:|---:|
| BBC | 160 | C-I | sensitivity_max | +0.109184 | -1.175337 |
| BBC | 160 | V-I | sensitivity_max | +0.244059 | -0.006423 |
| BBC | 400 | C-I | sensitivity_min | +0.064226 | -1.201028 |
| Reuters | 160 | C-I | sensitivity_max | +0.095238 | -0.071429 |

## CiteSeer Q400既定反例

| 政策 | 控制 | ACC | NMI | ARI | 训练BER | 池外BER |
|---|---|---:|---:|---:|---:|---:|
| primary_default | C-I | +0.099223 | -0.545820 | -0.042214 | +0.127584 | +0.442982 |
| sensitivity_min | C-I | +0.047455 | -0.639020 | -0.061368 | +0.324952 | +0.435037 |
| sensitivity_max | C-I | +0.056083 | -0.501296 | +0.002412 | +0.127584 | +0.328187 |
| primary_default | V-I | +0.276100 | -0.928022 | -0.071223 | -10.519100 | +0.529020 |
| sensitivity_min | V-I | +0.215703 | -0.929152 | -0.095294 | -10.190153 | +0.463965 |
| sensitivity_max | V-I | +0.284728 | -0.534722 | +0.090433 | -11.444696 | +0.210896 |

五指标差值均为100乘Full减控制，BER正差表示更差。all_ACC_positive_metric_tradeoffs.csv列出全部ACC正均值但NMI、ARI或池外BER变差的组，避免只呈现一个有利例子。

主文可用已核验的 text_and_tables/sensitivity_results_en.tex 和 _zh.tex；SI优先用本目录40行并排ACC表。全部600组五指标及6000个配对值继续保存在evaluation，不需全部放入正文。
