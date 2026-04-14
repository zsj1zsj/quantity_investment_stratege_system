# 量化投资决策系统 v2 (ML + 策略融合版)

基于 LightGBM + Specification Pattern 的量化投资决策系统，面向 S&P 500、NASDAQ Composite 及 7 个 Sector ETF 的多标的组合，以风险调整后收益为核心目标。

## 当前进度

### Phase 1：基础验证 — 已通过

| Gate 标准 | 要求 | SP500 | NASDAQ | 状态 |
|-----------|------|-------|--------|------|
| 信号质量 (prob>0.7 均值) | >0.5% | 0.967% | 1.175% | PASS |
| 回测年化收益 (扣成本) | >0% | +5.10% | +7.80% | PASS |
| 最大回撤 | <30% | 13.26% | 15.57% | PASS |

### Phase 2：信号增强与策略优化 — 已完成（Gate 2 全部通过）

**多标的组合（SP500+NASDAQ）— Gate 2 达标配置：**

| Gate 标准 | 要求 | MULTI | 状态 |
|-----------|------|-------|------|
| 年化收益 (扣成本) | >5% | 9.20% | PASS |
| 最大回撤 | <25% | 20.50% | PASS |
| 夏普比率 | >0.5 | 0.52 | PASS |
| 年均交易 | <50 | 11.7 | PASS |

**单标的回测（参考）：**

| 指标 | SP500 | NASDAQ |
|------|-------|--------|
| 年化收益 | 3.95% | 6.49% |
| 最大回撤 | 11.48% | 14.86% |
| 夏普比率 | -0.01 | 0.27 |
| 交易/年 | 5.1 | 6.2 |

**组合回测统计：**
- 总交易: 162笔 (11.7笔/年), 胜率69.8%, 盈亏比2.10
- 回测区间: 2012-02 ~ 2025-12 (约14年)

### Phase 3：组合扩展（10标的组合） — 已完成

**Gate 2 最终结果（10标的组合）：**

| Gate 标准 | 要求 | 10标的组合 | 9标的组合 | 状态 |
|-----------|------|-----------|----------|------|
| 年化收益 | >5% | 8.79% | 9.34% | PASS |
| 最大回撤 | <25% | 23.17% | 21.45% | PASS |
| 夏普比率 | >0.5 | 0.55 | **0.66** | PASS |
| 年均交易 | <50 | 25.1 | 24.8 | PASS |

**组合标的:** SP500, NASDAQ, NASDAQ100 + 7 Sector ETFs (XLK/XLF/XLV/XLI/XLC/XLY/XLP)
- NASDAQ100（^NDX）新增：AR=7.24%, DD=14.56%, SR=0.37，单标的优于 SP500
- NASDAQ100 与 NASDAQ 高相关，加入后 DD 略升（21.45%→23.17%），SR 略降（0.66→0.55）
- XLE (能源) 已排除：单标的 DD=33%, SR=-0.30，拖累组合表现
- 总暴露上限: 90%, 单标的上限: 30%

### 各标的独立性能分析（sector-analysis）

各资产独立回测结果（Walk-Forward，扣成本，按 SR 排序）：

| 标的 | AR | DD | SR | WR | AvgRet | 评级 | 说明 |
|------|-----|------|------|------|--------|------|------|
| NASDAQ | 6.96% | 13.66% | 0.32 | 68.2% | 1.50% | B | 综合最优 |
| XLV | 5.52% | 9.92% | 0.26 | 69.6% | 1.37% | B | 医疗，低回撤 |
| XLF | 6.17% | 14.16% | 0.24 | 72.7% | 1.41% | B | 金融，胜率最高 |
| XLY | 5.79% | 13.63% | 0.20 | 66.7% | 1.61% | B | 可选消费 |
| XLK | 5.77% | 14.51% | 0.17 | 65.6% | 1.51% | B | 科技 |
| XLC | 4.36% | 20.48% | 0.04 | 63.9% | 0.67% | B | 通信，回撤偏高 |
| XLI | 4.05% | 10.90% | 0.01 | 65.3% | 1.35% | B | 工业 |
| SP500 | 3.74% | 12.45% | -0.04 | 71.6% | 1.40% | D | 单独跑不赢无风险 |
| XLP | 2.98% | 10.95% | -0.21 | 65.6% | 0.91% | D | 必选消费，最弱 |

- 没有单标的达到 Grade A（SR>0.5）
- 组合后 SR=0.66 远高于任何单标的 → 分散化是关键收益来源
- SP500、XLP 单独表现弱，但仍留在组合中以提供相关性分散

### 稳定性分析

**逐年表现（2012-2025，9标的组合）：**

| 年份 | AR | DD | SR | 状态 |
|------|-----|------|------|------|
| 2012 | 7.20% | 6.72% | 0.40 | OK |
| 2013 | 23.76% | 1.18% | 3.22 | OK |
| 2014 | 12.93% | 2.86% | 1.47 | OK |
| 2015 | 7.89% | 6.31% | 0.59 | OK |
| 2016 | 7.82% | 7.90% | 0.54 | OK |
| 2017 | 12.58% | 0.93% | 1.92 | OK |
| 2018 | 3.36% | 11.81% | -0.08 | WEAK |
| 2019 | 12.36% | 2.40% | 1.29 | OK |
| 2020 | 1.20% | 11.55% | -0.22 | WEAK |
| 2021 | 26.12% | 2.94% | 2.42 | OK |
| 2022 | -16.66% | 21.45% | -1.93 | WEAK |
| 2023 | 23.89% | 4.11% | 2.21 | OK |
| 2024 | 26.65% | 1.71% | 3.24 | OK |
| 2025 | -6.83% | 13.50% | -1.28 | WEAK |

- Rolling 1Y Sharpe: mean=0.98, std=1.56
- 10/14年正收益（比2标的的9/14改善）
- 2022年仍是最大弱点（利率冲击）

**参数鲁棒性（9标的 vs 2标的）：**

| 参数 | 通过范围(9标的) | 通过范围(2标的) | 评估 |
|------|---------------|---------------|------|
| vix_caution | 15-22 全通过 | 15-22 全通过 | 极稳定 |
| vix_stress | 25-30 全通过 | 仅25通过 | **大幅改善** |
| hold_days | 20-35 全通过 | 仅25通过 | **大幅改善** |
| stop_loss | 6-15% 全通过 | 10-15% 通过 | 改善 |
| buy_thresh | 40-60% 全通过 | 仅50% 通过 | **大幅改善** |

---

## 系统架构

```
数据层 → 特征工程 → ML模型层 → 信号层 → 策略层 → 回测层 → 输出层
   ↑                                     ↑    ↑
VIX/TNX                            Regime检测  交易成本模型
跨市场数据                         (VIX-based)  Specification Pattern
```

## 安装

```bash
pip install yfinance pandas numpy ta lightgbm scikit-learn joblib pyarrow
```

## 使用

```bash
# 1. 下载数据（含 VIX、美债收益率、Sector ETFs）
python main.py fetch

# 2. 训练模型（LightGBM, 500/60 滑动窗口）
python main.py train

# 3. 信号质量验证（分桶分析, Gate 1 前置检查）
python main.py validate-signal

# 4. 回测（单标的 + 多标的组合，含交易成本）
python main.py backtest

# 5. 行业分析（各标的独立回测，评级 A/B/C/D）
python main.py sector-analysis

# 6. 稳定性分析（Rolling Sharpe/DD + 参数敏感性）
python main.py stability

# 7. 生成预测报告（JSON + 可读格式）
python main.py predict
```

## 项目结构

```
config.py                      # 全局参数配置
data/
  fetcher.py                   # Yahoo Finance 数据下载（含 VIX/TNX）
  store.py                     # Parquet 缓存管理
features/
  technical.py                 # 6层特征工程（趋势/动量/波动率/经典指标/成交量/跨市场）
  label.py                     # 标签生成（20日收益 > 3%）
model/
  train.py                     # LightGBM 滑动窗口训练
  evaluate.py                  # 评估指标（AUC/Precision/Recall/F1）
  predict.py                   # 预测 + 策略层决策输出
signal_layer/
  calibration.py               # Platt Scaling 校准 + EWM 平滑
strategy/
  spec.py                      # Specification Pattern（可组合交易规则）
  engine.py                    # 策略引擎（固定持仓期 + 仓位控制）
backtest/
  cost_model.py                # 交易成本模型（0.05%佣金 + 0.05%滑点/每边）
  engine.py                    # Walk-forward 回测引擎（单标的）
  multi_asset.py               # 多标的组合回测引擎
  sector_analysis.py           # 各标的独立性能分析（AR/DD/SR/WR，评级）
  signal_validation.py         # 信号质量分桶验证
  stability.py                 # Rolling 分析 + 参数敏感性测试
output/
  report.py                    # JSON + 可读报告输出
experiments/
  label_search.py              # 标签定义搜索实验
  strategy_search.py           # 策略参数搜索实验
main.py                        # CLI 入口
```

## v1 → v2 主要变更记录

### 模型层
| 项目 | v1 | v2 |
|------|----|----|
| 模型 | XGBoost | LightGBM |
| 标签 | 5日收益 > 0% | 20日收益 > 3% |
| 训练窗口 | 8年（日历年） | 500交易日 |
| 测试窗口 | 1年 | 60交易日 |
| 步长 | 1年 | 60交易日 |
| 特征数 | 13 | 21（含跨市场因子） |

### 新增特征
- **跨市场因子**: VIX绝对值、VIX 5日变化率、美债10年期收益率、利率5日变化、VIX×return_5d交互项
- **动量**: return_5d, return_10d, momentum_accel (动量加速度)
- **波动率**: vol_ratio (5d/20d波动率比)
- **趋势**: ma5_ma20_ratio (短长期趋势交叉)
- **成交量**: volume_zscore_20d (成交量20日Z-score)

### 特征重要性排名（训练结果）
1. TNX（美债收益率）— 最重要
2. VIX（波动率指数）
3. MACD Signal
4. Volatility 20d
5. MACD Histogram

### 新增模块
- **信号层** (`signal_layer/`): Platt Scaling 校准 + EWM 平滑
- **策略层** (`strategy/`): Specification Pattern 可组合规则，固定20日持仓
- **回测层** (`backtest/`): Walk-forward 回测，含交易成本、信号分桶验证
- **实验** (`experiments/`): 标签搜索、策略参数搜索

### 策略层设计
- **入场**: prob_up > 0.5（Regime 检测替代趋势过滤）
- **出场**: 持仓满25个交易日 或 止损 -10%
- **仓位**: prob > 0.8 → 100%, prob > 0.5 → 80%, per-asset cap 60%
- **Regime**: VIX < 18 正常, 18-25 减仓50%, ≥25 禁止入场
- **多标的**: SP500 + NASDAQ 组合管理，分散化提升夏普
- **交易成本**: 每笔 0.2%（买+卖）

### 迭代过程
1. **改标签** (5d>1% → 20d>3%): Gate 1 从 FAIL 转为 PASS
2. **引入跨市场因子** (VIX/TNX): NASDAQ 信号质量提升19%
3. **加成交量特征** (volume_zscore_20d): 微幅提升
4. **删零贡献特征** (price_up_vol_down等): 减少噪声
5. **固定20日持仓**: 回测年化从 -0.22% 转为 +1.22%（SP500）
6. **开启TrendUp过滤**: 回撤从34.9%降至12.76%
7. **VIX Regime替代TrendUp**: 更好的下行保护，收益不受损
8. **提升仓位至80/100%**: 年化从3%级提升至5%+
9. **收紧VIX阈值** (20/30 → 18/25): 回撤从27%降至20%
10. **多标的组合** (SP500+NASDAQ): 夏普从0.14/0.36提升至0.52

## 输出示例

```json
{
  "date": "2026-04-08",
  "symbol": "SP500",
  "signal_strength": "very_low",
  "probability": 0.2503,
  "suggestion": "HOLD",
  "position_size": 0.0,
  "regime": "normal",
  "risk_note": "vol_ratio=0.78, volatility normal",
  "model_version": "v2.0-lgbm"
}
```

## 注意事项

- 数据源为 Yahoo Finance，频繁请求可能触发限流，遇到报错等几分钟再试
- 预测结果仅供参考，不构成投资建议
- 更新数据后需重新训练模型才能反映最新市场状态
- 回测使用样本外数据（Walk-forward），但仍存在过拟合风险，不应对结果过度自信
