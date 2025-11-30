import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:\pg\course/bigdata\Scheduling-Advertising-on-Cable-Television\output\deal_completion.csv')
df['deal_id'] = df['deal_id'].astype(str)  # 确保是字符串

plt.figure(figsize=(14, 6))
bars = plt.bar(range(len(df)), df['completion'])  # 🔹 使用 index 代替 Deal 标签

for i, v in enumerate(df['completion']):
    if v > 0.5:
        bars[i].set_color('green')
    elif v > 0.3:
        bars[i].set_color('yellow')
    elif v > 0.1:
        bars[i].set_color('orange')
    else:
        bars[i].set_color('red')

plt.xlabel('Deal')
plt.ylabel('Completion Rate')
plt.title('Completion Performance of All Deals')
plt.ylim(0, 1)
plt.xticks(rotation=90)  # 🔥横轴调整
plt.tight_layout()       # 防止文字被挤掉
plt.show()


###
# 
# 去除为0的
###
# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取数据
# df = pd.read_csv('D:\pg\course/bigdata\Scheduling-Advertising-on-Cable-Television\output\deal_completion.csv')

# # 过滤掉 completion_rate 为 0 的 deal
# df_nonzero = df[df['completion'] > 0].reset_index(drop=True)

# plt.figure(figsize=(14, 6))
# bars = plt.bar(range(len(df_nonzero)), df_nonzero['completion'])

# # 颜色设置
# for i, v in enumerate(df_nonzero['completion']):
#     if v > 0.5:
#         bars[i].set_color('green')
#     elif v > 0.3:
#         bars[i].set_color('yellow')
#     elif v > 0.1:
#         bars[i].set_color('orange')
#     else:
#         bars[i].set_color('red')

# plt.ylabel('Completion Rate')
# plt.title('Completion Performance of Deals (Excluding 0 Completion)')
# plt.ylim(0, 1)

# plt.xticks([])  # 🔕 隐藏横轴
# plt.tight_layout()
# plt.show()
