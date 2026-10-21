import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\alarm_records.csv")

df["trigger_time"] = pd.to_datetime(df["trigger_time"])
df["heartbeat_synced_at"] = pd.to_datetime(df["heartbeat_synced_at"])

# 计算触发时刻与心跳最后同步时刻的时间差（秒）
df["heartbeat_lag_sec"] = (df["trigger_time"] - df["heartbeat_synced_at"]).dt.total_seconds()

print("=== 关键：heartbeat_ttl_sec 与 is_valid 的关系 ===")
print("ttl=0 的数量:", (df["heartbeat_ttl_sec"]==0).sum())
print("ttl=0 中 is_valid 分布:")
print(df[df["heartbeat_ttl_sec"]==0]["is_valid"].value_counts(dropna=False))
print()
print("ttl>0 中 is_valid 分布:")
print(df[df["heartbeat_ttl_sec"]>0]["is_valid"].value_counts(dropna=False))
print()

# 有闭环核验的样本
closed = df[df["is_valid"].notna()].copy()
print("=== 有闭环样本按 ttl 分组 ===")
print("ttl=0:", len(closed[closed["heartbeat_ttl_sec"]==0]), "误报率 =", round((closed[closed["heartbeat_ttl_sec"]==0]["is_valid"]==0).mean(),3))
print("ttl>0:", len(closed[closed["heartbeat_ttl_sec"]>0]), "误报率 =", round((closed[closed["heartbeat_ttl_sec"]>0]["is_valid"]==0).mean(),3))
print()

# 分桶看 ttl 与误报率
closed["ttl_bin"] = pd.cut(closed["heartbeat_ttl_sec"], bins=[-1,0,5,10,15,20,25,30], labels=["0","1-5","6-10","11-15","16-20","21-25","26-30"])
print("=== 各 ttl 区间误报率 ===")
tb = closed.groupby("ttl_bin").agg(total=("is_valid","size"), false_rate=("is_valid", lambda x: round((x==0).mean(),3)))
print(tb)
print()

# heartbeat_lag 与 ttl 的关系
print("=== lag 统计（秒）===")
print(df["heartbeat_lag_sec"].describe())
print()
print("=== ttl=0 样本的 lag 分布 ===")
print(df[df["heartbeat_ttl_sec"]==0]["heartbeat_lag_sec"].describe())
print()

# service_type 维度误报率
print("=== 各 service_type 误报率（有闭环）===")
st = closed.groupby("service_type").agg(total=("is_valid","size"), false_cnt=("is_valid", lambda x:(x==0).sum()))
st["false_rate"] = round(st["false_cnt"]/st["total"],3)
print(st.sort_values("false_rate", ascending=False))
print()

# monitor_node 维度误报率
print("=== 各 monitor_node 误报率 ===")
mn = closed.groupby("monitor_node").agg(total=("is_valid","size"), false_cnt=("is_valid", lambda x:(x==0).sum()))
mn["false_rate"] = round(mn["false_cnt"]/mn["total"],3)
print(mn)
print()

# concurrent_qps 与误报的关系
print("=== qps 与误报 ===")
print("误报样本 qps 均值:", round(closed[closed["is_valid"]==0]["concurrent_qps"].mean(),2))
print("真实样本 qps 均值:", round(closed[closed["is_valid"]==1]["concurrent_qps"].mean(),2))
