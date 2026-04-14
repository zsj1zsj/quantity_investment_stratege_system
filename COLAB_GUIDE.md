# Google Colab 使用手册

Google Colab 免费提供 GPU/CPU 运算环境，但**会话断开后所有文件会消失**。
本手册通过挂载 Google Drive 解决数据和模型的持久化问题。

---

## 一、首次setup（只需做一次）

### Cell 1 — 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

> 点击弹出的授权链接，允许 Colab 访问你的 Drive。

---

### Cell 2 — 克隆代码库

```python
import os

REPO_DIR = '/content/drive/MyDrive/quantity_invest'

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/zsj1zsj/quantity_investment_stratege_system.git {REPO_DIR}
    print("首次克隆完成")
else:
    # 已存在则拉取最新代码
    !cd {REPO_DIR} && git pull
    print("代码已更新")

os.chdir(REPO_DIR)
!pwd
```

> 代码克隆到 Drive，之后重开会话无需再克隆。

---

### Cell 3 — 安装依赖

```python
!pip install -q yfinance pandas numpy ta lightgbm scikit-learn joblib pyarrow
print("依赖安装完成")
```

> Colab 每次新会话都需要重新安装，约 1 分钟。

---

### Cell 4 — 验证环境

```python
import sys
sys.path.insert(0, REPO_DIR)

import lightgbm, yfinance, ta
print(f"LightGBM: {lightgbm.__version__}")
print(f"yfinance: {yfinance.__version__}")
print("环境就绪 ✓")
```

---

## 二、下载数据（首次 或 每月更新一次）

### Cell 5 — 下载全部数据

```python
os.chdir(REPO_DIR)
!python main.py fetch
```

**耗时约 3-5 分钟**，下载 13 个标的从 2010 年至今的日线数据，保存在：
```
/content/drive/MyDrive/quantity_invest/data/cache/
```

遇到限流报错（`YFRateLimitError`）时等 2-3 分钟重试：

```python
import time
time.sleep(180)
!python main.py fetch
```

> **数据保存在 Drive，下次开会话无需重新下载。**

---

## 三、训练模型（首次 或 每月更新一次）

### Cell 6 — 训练

```python
os.chdir(REPO_DIR)
!python main.py train
```

**耗时约 5-10 分钟**，输出每个滑动窗口的 AUC 指标，模型保存在：
```
/content/drive/MyDrive/quantity_invest/model/saved/
```

> **模型保存在 Drive，下次开会话无需重新训练。**

---

## 四、日常使用（每周操作）

每次开启新会话，只需执行以下 setup cells，然后直接到预测步骤：

```python
# 每次新会话必跑（约 1 分钟）
from google.colab import drive
drive.mount('/content/drive')

import os, sys
REPO_DIR = '/content/drive/MyDrive/quantity_invest'
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

!pip install -q yfinance pandas numpy ta lightgbm scikit-learn joblib pyarrow
!cd {REPO_DIR} && git pull   # 拉取代码更新
print("就绪 ✓")
```

---

### Cell 7 — 生成今日预测信号

```python
os.chdir(REPO_DIR)
!python main.py predict
```

**示例输出：**
```
============================================================
  Quantitative Investment Decision Report - 2026-04-14
============================================================

  S&P 500 (^GSPC)
  Last Close: 5203.58  (2026-04-11)
  Signal Strength: very_low
  Probability (20-day >3%): 25.03%
  Suggestion: HOLD
  Position Size: 0%
  Regime: caution
  Risk Note: vol_ratio=0.78, volatility normal
```

**信号解读：**

| Suggestion | Regime | 操作建议 |
|------------|--------|----------|
| BUY | normal | 按 Position Size × 30% 建仓 |
| BUY | caution | 减半建仓，VIX 偏高谨慎 |
| BUY | stress | 忽略，VIX≥25 不入场 |
| HOLD | 任意 | 不操作 |

---

### Cell 8（可选）— 更新数据后重新预测

如果距上次 fetch 超过一周，先更新数据再预测：

```python
os.chdir(REPO_DIR)
!python main.py fetch   # 更新数据
!python main.py predict # 重新预测
```

> 注意：更新数据后**不一定需要重新训练**，除非过了一个月。

---

## 五、深度分析（按需运行）

### 查看完整回测

```python
os.chdir(REPO_DIR)
!python main.py backtest
```

> 耗时约 15-30 分钟（9 个标的各自训练），了解策略历史表现用。

---

### 样本外验证（过拟合检查）

```python
os.chdir(REPO_DIR)
!python main.py holdout
```

> 对比 2012-2023 样本内 vs 2024-2026 样本外表现，判断策略稳健性。

---

### 行业分析

```python
os.chdir(REPO_DIR)
!python main.py sector-analysis
```

> 查看各行业 ETF 的独立胜率和收益，了解哪个行业信号质量更高。

---

### 稳定性分析

```python
os.chdir(REPO_DIR)
!python main.py stability
```

> 逐年表现 + 参数敏感性测试，判断策略在不同参数下是否稳健。

---

## 六、完整会话模板（复制此模板直接使用）

以下是一个完整 Colab notebook 的 cell 顺序，**每周操作**只需执行前 3 个 cell：

```python
# ===== Cell 1: 环境初始化（每次必跑，约 1 分钟）=====
from google.colab import drive
drive.mount('/content/drive')

import os, sys
REPO_DIR = '/content/drive/MyDrive/quantity_invest'
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

!pip install -q yfinance pandas numpy ta lightgbm scikit-learn joblib pyarrow
!cd {REPO_DIR} && git pull
print("就绪 ✓")
```

```python
# ===== Cell 2: 更新数据（每周一次）=====
!python main.py fetch
```

```python
# ===== Cell 3: 生成预测（核心）=====
!python main.py predict
```

```python
# ===== Cell 4: 重新训练（每月一次，可选）=====
!python main.py train
```

---

## 七、常见问题

**Q: `YFRateLimitError` 下载失败？**
A: Yahoo Finance 有请求频率限制，等待 2-3 分钟后重跑 `fetch`。

**Q: 重开会话后提示找不到模型？**
A: 确认 Cell 1 中已挂载 Drive 且 `REPO_DIR` 路径正确。模型应在 `model/saved/` 下。

**Q: 预测只显示 SP500 和 NASDAQ，没有 ETF？**
A: ETF 预测目前不在 `predict` 命令中，需通过 `sector-analysis` 或 `backtest` 查看。

**Q: Colab 免费版够用吗？**
A: 够用。`fetch` 和 `predict` 不需要 GPU，`train` 用 CPU 也能完成，只是稍慢。

**Q: 数据多久更新一次合适？**
A: 每周更新数据 + 预测。模型每月重训练一次即可（15 分钟内完成）。
