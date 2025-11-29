import pandas as pd

# ========== 1. 读取文件（路径已修复） ==========
deals = pd.read_csv('D:\pg\course/bigdata\Scheduling-Advertising-on-Cable-Television\data\deals_stage1.csv')
schedule = pd.read_csv('D:\pg\course/bigdata\Scheduling-Advertising-on-Cable-Television\output\stage2_schedule.csv')
ratings = pd.read_csv('D:\pg\course/bigdata\Scheduling-Advertising-on-Cable-Television\data/ratings_stage2.csv')

# ========== 2. 只保留已排期广告 ==========
schedule = schedule[schedule['status'].str.lower() == "scheduled"].copy()

# ========== 3. 重命名 demo 列 ==========
ratings = ratings.rename(columns={'demo_id': 'target_demo'})

# ========== 4. 拼接收视率 ==========
merged = schedule.merge(ratings, on=['break_id', 'target_demo'], how='left')

# 检查是否有缺 rating 的情况（提醒用）
missing_ratings = merged[merged['rating'].isna()]
if len(missing_ratings) > 0:
    print("⚠️ 有广告没有匹配到收视率 (rating)，例如：")
    print(missing_ratings[['break_id', 'ad_id', 'target_demo']].head())

# ========== 5. 计算曝光量 ==========
merged['impressions'] = (merged['length_sec'] / 30) * merged['rating']

# ========== 6. 聚合到 deal ==========
delivered = merged.groupby('deal_id')['impressions'].sum().reset_index()

# ========== 7. 拼合合同信息（I_d） ==========
result = deals.merge(delivered, on='deal_id', how='left')
result['impressions'] = result['impressions'].fillna(0)

# ========== 8. 计算完成情况 ==========
result['completion'] = result['impressions'] / result['I_d']
result['shortfall'] = result['I_d'] - result['impressions']
result['completion'] = result['completion'].clip(upper=1)
result['shortfall'] = result['shortfall'].clip(lower=0)

# ========== 9. 排序 & 输出 ==========
result_sorted = result.sort_values('completion', ascending=False)

print("📊 成交完成情况（Top 10）：")
print(result_sorted.head(10))

# 你也可以保存为 CSV
result_sorted.to_csv('D:\pg\course/bigdata\Scheduling-Advertising-on-Cable-Television\output\deal_completion.csv', index=False)
print("💾 结果已保存至 output/deal_completion.csv")

