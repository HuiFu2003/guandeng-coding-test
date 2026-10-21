import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\alarm_records.csv")
df["trigger_time"] = pd.to_datetime(df["trigger_time"])
df["heartbeat_synced_at"] = pd.to_datetime(df["heartbeat_synced_at"])
df["heartbeat_lag_sec"] = (df["trigger_time"] - df["heartbeat_synced_at"]).dt.total_seconds()
closed = df[df["is_valid"].notna()].copy()

# 特征工程：加入 lag 与 ttl 的交互、节点、服务
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

X = closed[["heartbeat_lag_sec", "heartbeat_ttl_sec", "concurrent_qps"]]
# 加入 one-hot
X = pd.concat([X, pd.get_dummies(closed["monitor_node"], prefix="node"),
                  pd.get_dummies(closed["service_type"], prefix="svc")], axis=1)
y = (closed["is_valid"]==0).astype(int)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
print("决策树 depth=3 交叉验证 F1 (预测误报):", round(scores.mean(),3), "+-", round(scores.std(),3))

clf.fit(X, y)
print()
print("=== 决策树特征重要性 top ===")
imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(8))
print()
print("=== 树结构 ===")
from sklearn.tree import plot_tree, export_text
print(export_text(clf, feature_names=list(X.columns)))
