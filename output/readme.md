Stage 2: 每日广告精排模型 (Daily Ad Scheduling Optimization)
1. 模型简介
Stage 2 是广告排期系统的核心优化引擎，负责每日的广告精细化排期。这份Notebook 实现了一个混合整数规划（MIP）模型，它接收来自 Stage 1 的战略性指导，并结合当日的详细运营数据，最终生成一个最优的24小时广告排期方案。
核心目标:
最大化加权曝光价值: 优先将收视率高的时段分配给在 Stage 1 中被判定为更重要、更紧急的广告合约（即 W_d 值高的合约）。
满足复杂业务约束: 严格遵守广告主提出的各种排期规则，如广告分离、品牌分散、特殊位置（A/Z-position）、关联播出（Piggyback/Sandwich）等。
生成可行排期: 产出一个无冲突的、广告到时段的分配方案，可直接用于后续 Stage 3（时段内排序）的处理。
2. 
准备文件: 准备好由 Stage 1 生成的 stage1_weights.csv 和 stage1_xdb.csv 文件
下载结果: 脚本运行成功后， output/ 目录下会生成一个新的 stage2_schedule.csv 文件。
3. 输入文件说明
需准备并上传由 Stage 1 生成的 2 个 CSV 文件。
确保上传后的文件路径为：
/content/output/stage1_weights.csv
/content/output/stage1_xdb.csv
该文件提供了每个广告合约的战略重要性。
列名	类型	说明
deal_id	string	合约的唯一 ID。
W_d	float	核心输入。合约的权重或优先级，将直接用作 Stage 2 目标函数中的系数，决定了合约的价值。
该文件提供了周度的粗排计划，用于为每日优化提供一个高质量的起点。
列名	类型	说明
deal_id	string	合约的唯一 ID。
break_id	string	时段 ID。
x_db	int	（或 num_ads）分配到该时段的广告数量。
用途	-	用于生成一个初始解（Warm Start），这能极大地帮助求解器更快地找到最优或高质量的可行解。
4. Notebook 生成及输出的文件
本 Notebook 的代码可以（已取消生成文件）在 data/ 目录下创建以下模拟的当日数据文件：
data/deals_stage2.csv: 当日所有待播广告的详细信息。
data/breaks_stage2.csv: 当日所有可用广告时段的详细库存信息。
data/ratings_stage2.csv: 各时段对不同人群的预测收视率。
运行成功后，Notebook 将在 output/ 目录下生成一个新的排期文件：
一个内容详尽的 CSV 文件，展示了最终的广告排期结果，可以直接用于数据分析或后续流程。
列名	类型	说明
break_id	string	广告被分配到的时段 ID。如果未被排期，则显示为 "BINNED"。
ad_id	string	广告的唯一 ID。
deal_id	string	该广告所属的合约 ID。
length_sec	int	广告时长（秒）。
target_demo	string	广告的目标受众。
advertiser	string	广告主名称。
brand	string	品牌名称。
category	string	品牌所属品类。
status	string	Scheduled (已排期) 或 Binned (放入回收站，未排期)。