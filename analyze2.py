import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\alarm_records.csv")
df["trigger_time"] = pd.to_datetime(df["trigger_time"])
df["heartbeat_synced_at"] = pd.to_datetime(df["heartbeat_synced_at"])
df["heartbeat_lag_sec"] = (df["trigger_time"] - df["heartbeat_synced_at"]).dt.total_seconds()

closed = df[df["is_valid"].notna()].copy()

# lag 分桶看误报率
closed["lag_bin"] = pd.cut(closed["heartbeat_lag_sec"], bins=[-1,60,300,600,1200,1800,2400,3000,4000],
    labels=["0-60s","1-5m","5-10m","10-20m","20-30m","30-40m","40-50m","50m+"])
print("=== heartbeat_lag 分桶误报率 ===")
lb = closed.groupby("lag_bin").agg(total=("is_valid","size"), false_cnt=("is_valid", lambda x:(x==0).sum()))
lb["false_rate"] = round(lb["false_cnt"]/lb["total"],3)
print(lb)
print()

# 关键：ttl 与 lag 的组合。ttl 本质 = 心跳有效剩余秒数
# 如果 heartbeat_synced_at + ttl 秒 之后，心跳还没到，就是"滞后心跳"
# 计算：心跳"应到时间" = synced_at + ttl，若 trigger_time > 应到时间 说明 TTL 已耗尽（滞后）
df["ttl_deadline"] = df["heartbeat_synced_at"] + pd.to_timedelta(df["heartbeat_ttl_sec"], unit="s")
df["lag_overdue"] = (df["trigger_time"] > df["ttl_deadline"]).astype(int)
print("=== ttl 已耗尽(滞后心跳) 数量 ===")
print(df["lag_overdue"].value_counts())
print()
c2 = closed.copy()
print("=== 滞后心跳 vs 误报 ===")
print("滞后(overdue):", (c2["lag_overdue"]==1).sum(), "误报率 =", round((c2[c2["lag_overdue"]==1]["is_valid"]==0).mean(),3))
print("未滞后:", (c2["lag_overdue"]==0).sum(), "误报率 =", round((c2[c2["lag_overdue"]==0]["is_valid"]==0).mean(),3))
print()

# 但注意：heartbeat_synced_at 本身可能就是滞后很久的旧心跳
# 更本质：lag_sec 越大越像滞后心跳。用阈值实验
print("=== 不同 lag 阈值的误报率（lag > 阈值 判定为滞后）===")
for th in [300, 600, 900, 1200, 1800, 2400]:
    pred_lag = (c2["heartbeat_lag_sec"] > th).astype(int)
    fp = ((pred_lag==1) & (c2["is_valid"]==0)).sum()
    fn = ((pred_lag==0) & (c2["is_valid"]==0)).sum()
    tp = ((pred_lag==1) & (c2["is_valid"]==1)).sum()
    precision = fp/(pred_lag.sum()) if pred_lag.sum()>0 else 0
    print(f"  lag>{th}s: 标记滞后数={pred_lag.sum()}, 其中误报={fp}, 精确率(标记中真误报占比)={round(precision,3)}, 漏报误报={fn}")

# 关键交叉：ttl 小 AND lag 大
print()
print("=== ttl<=5 且 lag>600 的组合 ===")
mask = (closed["heartbeat_ttl_sec"]<=5) & (closed["heartbeat_lag_sec"]>600)
print("样本数:", mask.sum(), "误报率:", round((closed[mask]["is_valid"]==0).mean(),3))
print("ttl<=10 且 lag>300:")
mask2 = (closed["heartbeat_ttl_sec"]<=10) & (closed["heartbeat_lag_sec"]>300)
print("样本数:", mask2.sum(), "误报率:", round((closed[mask2]["is_valid"]==0).mean(),3))
