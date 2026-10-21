import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\alarm_records.csv")
df["trigger_time"] = pd.to_datetime(df["trigger_time"])
df["heartbeat_synced_at"] = pd.to_datetime(df["heartbeat_synced_at"])
df["heartbeat_lag_sec"] = (df["trigger_time"] - df["heartbeat_synced_at"]).dt.total_seconds()

closed = df[df["is_valid"].notna()].copy()

# 口径成本：每次误报排查 75 分钟；每次提前标记滞后心跳 10 分钟
COST_FALSE = 75   # 误报排查损失(分钟)
COST_TAG   = 10   # 提前标记滞后心跳(分钟)

print("=== 成本收益分析：不同 lag 阈值下，标记滞后心跳的净收益 ===")
print(f"口径：误报排查 {COST_FALSE} 分钟/次；标记滞后心跳 {COST_TAG} 分钟/次")
print()
print(f"{'阈值':<10}{'标记数':<8}{'其中误报':<10}{'漏误报':<8}{'精确率':<8}{'净省时间(分钟)':<14}{'备注'}")
best = None
for th in [300, 420, 540, 600, 720, 900, 1080, 1200, 1500, 1800]:
    pred = (closed["heartbeat_lag_sec"] > th).astype(int)
    n_tag = pred.sum()
    fp = ((pred==1) & (closed["is_valid"]==0)).sum()  # 标记中确认为误报的
    fn = ((pred==0) & (closed["is_valid"]==0)).sum()  # 漏掉的误报
    # 收益：标记中误报省下排查时间；成本：标记数*10分钟
    gain = fp * COST_FALSE - n_tag * COST_TAG
    prec = fp/n_tag if n_tag>0 else 0
    note = ""
    if best is None or gain > best[1]:
        best = (th, gain, n_tag, fp, fn, prec)
    print(f"lag>{th:<5}{n_tag:<8}{fp:<10}{fn:<8}{round(prec,3):<8}{gain:<14}{note}")

print()
th, gain, n_tag, fp, fn, prec = best
print(f"=== 最优阈值 lag>{th}s ===")
print(f"标记滞后心跳 {n_tag} 次，其中确认误报 {fp} 次，漏报误报 {fn} 次")
print(f"净节省时间 = {fp}*{COST_FALSE} - {n_tag}*{COST_TAG} = {gain} 分钟 = {round(gain/60,1)} 小时")
print(f"精确率 = {round(prec,3)}")
print()

# 总误报量
total_false = (closed["is_valid"]==0).sum()
print(f"全量误报数 = {total_false}，漏报 {fn} 个，漏报率 {round(fn/total_false,3)}")
print()

# 额外维度：monitor_node 华东节点误报率极低(5.4%)，可能节点特征
# 综合规则：lag 阈值 + 节点
print("=== 组合规则：lag>540s 且 node!=华东 的误报率 ===")
m = (closed["heartbeat_lag_sec"]>540) & (closed["monitor_node"]!="华东节点")
print("样本数:", m.sum(), "误报率:", round((closed[m]["is_valid"]==0).mean(),3))

# service_type 高误报 + lag
print()
print("=== 高误报服务(search_engine/cache_node/dns_resolver) 且 lag>600 ===")
m2 = (closed["heartbeat_lag_sec"]>600) & (closed["service_type"].isin(["search_engine","cache_node","dns_resolver"]))
print("样本数:", m2.sum(), "误报率:", round((closed[m2]["is_valid"]==0).mean(),3))
