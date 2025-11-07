├─ stage1.py
├─ data/
│   ├─ deals_stage1.csv    # 合同数据（deals）
│   ├─ breaks_stage1.csv   # break 列表（库存）
│   └─ ratings_stage1.csv  # rating 预测（demo × break）
└─ output/
    ├─ stage1_weights.csv
    └─ stage1_xdb.csv

#数据说明

输入部分
1.deals_stage1.csv
必须字段
| 列名            | 类型     | 说明                                                                                           |
| ------------- | ------ | -------------------------------------------------------------------------------------------- |
| `deal_id`     | string | 合同 ID，唯一标识一个 deal，例如 `D1`、`D2`                                                               |
| `target_demo` | string | 目标受众群体（demographic），例如 `F18-34`, `M25-54`, `P25-54`，会去和 `ratings_stage1.csv` 里的 `demo_id` 对应 |
| `I_d`         | float  | 本周该 deal 期望交付的目标曝光量（impressions），即论文中的 (I_d)                                                 |
| `J_d`         | float  | 本周该 deal 最多允许播出的广告条数，即论文中的 (J_d)                                                             |
| `CPM_d`       | float  | 每 1000 人曝光的成本，用来给未满足曝光 (y_d) 定价                                                              |

可选字段
| 列名              | 示例            | 用途（可选）               |
| --------------- | ------------- | -------------------- |
| `client`        | `Retail_A`    | 客户名称，用于报表展示          |
| `priority`      | `high/low`    | 优先级，可以在后续版本用来加权或加约束  |
| `channel_pref`  | `C1` / `C1-2` | 偏好的频道，Stage1 里暂时没用   |
| `weekday_focus` | `Mon-Thu`     | 重点投放的星期，Stage1 里暂时没用 |

2.breaks_stage1.csv
必须字段
| 列名           | 类型     | 说明                              |
| ------------ | ------ | ------------------------------- |
| `break_id`   | string | break ID，例如 `B1`、`B2`           |
| `length_sec` | float  | 该 break 的可用广告总时长（秒），即论文中的 (L_b) |

可选字段
| 列名               | 示例                                | 用途（可选）      |
| ---------------- | --------------------------------- | ----------- |
| `channel`        | `C1` / `C2`                       | 频道 ID       |
| `day_of_week`    | `Mon` / `Tue`                     | 星期几         |
| `time_slot`      | `Daytime` / `Prime` / `LateNight` | 时段类型        |
| `inventory_type` | `Regular` / `Premium`             | 区分普通库存和黄金时段 |


3.ratings_stage1.csv
必须字段
| 列名         | 类型     | 说明                                   |
| ---------- | ------ | ------------------------------------ |
| `break_id` | string | break ID                             |
| `demo_id`  | string | demographic ID，需要与 `target_demo` 对得上 |
| `rating`   | float  | 该 demo 在该 break 的 rating 预测值         |   ------ 来源于ML，其他必须字段自编


输出部分
Stage 1 求解完成后，会在 `output/` 目录下生成两个 CSV 文件，作为后续 Stage 2 的输入和参考：

```text
output/
├─ stage1_weights.csv  # 每个 deal 的权重 W_d 和未满足曝光 y_d
└─ stage1_xdb.csv      # 一周粗排结果：x_{d,b}
