# 日化美妆产业链分析平台

> **L4 系统化答辩项目** — 基于多模型融合的日化美妆产业数据智能分析决策平台

---

## 🖥️ 快速开始

### 环境要求

```
Python 3.10+
```

### 安装依赖

```bash
pip install streamlit pandas numpy plotly openpyxl scikit-learn xgboost lightgbm joblib
```

### 准备数据文件

将以下三个 Excel 文件放在项目根目录：

| 文件 | 说明 |
|------|------|
| `tmall_order_report.xlsx` | 天猫订单报告 |
| `双十一淘宝美妆数据.xlsx` | 双十一淘宝美妆数据 |
| `日化.xlsx` | 日化销售订单 + 商品信息 |

### 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

---

## 📊 功能模块

| Tab | 模块 | 功能 |
|-----|------|------|
| 📋 | 数据概览 | 多数据源加载、基础统计、数据质量检查 |
| 🏷️ | 品牌竞争力 | 品牌 GMV 排名、价格-销量关系、品牌集中度、市场份额 |
| 🏭 | 渠道健康度 | 省份/地市覆盖分析、区域销售分布地图、渠道效率 |
| 🛒 | 零售效率 | 价格段分析、价格弹性测算、品类生命周期判断 |
| 👤 | 用户画像 | 客户价值分层 (RFM)、复购率分析、用户行为特征 |
| 🔗 | 联动洞察 | 退款预测 (XGBoost)、区域空白市场扫描 (ML 潜力预测) |
| 📊 | 销售策略 | 智能库存调整、商品热度预测、销量预测 |
| 🎁 | 套餐推荐 | K-Means 品类聚类 + 共现分析、套餐组合建议 |
| 💰 | 折扣策略 | 价格弹性模型融合、折扣效果模拟、利润/销量优化 |

---

## 🔧 技术架构

### 模型矩阵

| 模型 | 算法 | 用途 | 关键文件 |
|------|------|------|----------|
| 退款预测 | XGBoost 分类器 | 预测订单退款风险，辅助客服前置干预 | `models/refund_xgboost.joblib` |
| 价格弹性 | XGBoost 回归 | 预测不同折扣率下的销量变化 | `models/price_elasticity_best.joblib` |
| 销量预测 | XGBoost 回归 | 预估商品未来销量，指导库存备货 | `models/sales_forecast_xgboost_optimized.joblib` |
| 商品热度 | XGBoost 分类器 | 评估商品市场热度等级 | `models/product_hotness_best.joblib` |
| 区域潜力 | RandomForest 回归 | 预测商品在各地区的销售潜力 | `models/region_sales_potential_model.joblib` |
| 区域库存 | RandomForest 回归 | 区域最优库存水平预测 | `models/region_inventory_best.joblib` |
| 支付转化 | LogisticRegression | 预测订单支付转化概率 | `models/payment_conversion_best.joblib` |
| 品类聚类 | K-Means | 基于共现模式的商品品类聚类 | `models/product_clustering_model.joblib` |

### 模型对比

每个业务场景训练了多类模型并择优部署：

- **XGBoost** — 主力模型，树结构 + 梯度提升，适合表格数据
- **LightGBM** — 对比模型，直方图加速，训练更快
- **RandomForest** — 基线对比，Bagging 集成，抗过拟合
- **GradientBoosting** — 传统提升方法，作为对照
- **LogisticRegression / LinearRegression** — 线性基线
- **PyTorch MLP / LSTM** — 深度学习对照（.pt 文件）

### 数据流

```
Excel 数据源 (3个)
    │
    ├── tmall_order_report.xlsx ──→ 订单分析、退款预测、用户画像
    ├── 双十一淘宝美妆数据.xlsx ──→ 品牌分析、价格弹性、品类洞察
    └── 日化.xlsx ──→ 聚类分析、套餐推荐、区域潜力
    │
    ▼
数据预处理 (openpyxl + pandas)
    │
    ▼
特征工程 (价格段、时间特征、共现矩阵、RFM)
    │
    ▼
多模型训练 + 对比 (train/*.py)
    │
    ▼
模型择优部署 (models/*.joblib)
    │
    ▼
Streamlit 交互式分析 (app.py)
```

---

## 💡 创新点

### 1. 多模型融合决策
每个业务问题同时训练 6-8 种算法（XGBoost、LightGBM、RandomForest、GradientBoosting、线性模型、深度学习），自动对比指标后选择最优模型部署，而非单一模型套用所有场景。

### 2. 实时交互式 ML 推理
应用内所有图表和指标均基于真实数据 + 预训练模型实时计算，用户调节参数（折扣率、支持度阈值、库存水平）后，预测结果即时更新，而非静态报表展示。

### 3. 购买共现驱动的套餐推荐
构建品类共现矩阵（Co-occurrence Matrix），通过 K-Means 聚类发现购买行为相似的品类群，在簇内按共现强度生成套餐组合，替代传统的人工搭配。

### 4. 区域空白市场智能扫描
使用 RandomForest 回归模型预测商品在各地区的销售潜力，识别「有需求但无覆盖」的空白市场，为地推团队提供数据驱动的开荒建议。

### 5. 从数据到决策的闭环
平台覆盖「数据概览 → 诊断分析 → 预测建模 → 策略建议」完整链路，每个分析模块均输出可执行的商家赋能建议，而非停留在可视化展示。

---

## 🎯 解决的实际问题

| 业务问题 | 技术方案 | 实际效果 |
|----------|----------|----------|
| **退款率过高** | XGBoost 退款预测模型，识别高风险订单 | 提前干预可降低退款率，AUC > 0.85 |
| **折扣策略盲目** | 价格弹性模型 + 折扣效果模拟 | 量化折扣对销量/利润的影响，找到最优折扣点 |
| **库存积压或缺货** | 销量预测 + 热度分级 + 补货建议 | 动态调整库存，降低滞销风险 |
| **套餐搭配靠经验** | 共现矩阵 + K-Means 聚类 + 实时套餐生成 | 数据驱动的科学搭配，支持度量化 |
| **区域拓展无方向** | 区域潜力预测 + 空白市场扫描 | 精准识别高潜力未覆盖地区 |
| **商品热度难评估** | 多特征 XGBoost 热度分类器 | 自动化热度分级，指导资源倾斜 |
| **用户价值模糊** | RFM 模型 + 客户分层 | 识别高价值客户，精细化运营 |
| **品牌竞争力未知** | 市场份额矩阵 + 价格-销量象限分析 | 定位品牌在竞争格局中的位置 |

---

## 📁 项目结构

```
├── app.py                          # 主应用 (Streamlit)
├── app_backup.py                   # 早期备份
│
├── train/                          # 训练脚本
│   ├── payment_conversion_model.py
│   ├── price_elasticity_model.py
│   ├── product_hotness_model.py
│   ├── region_inventory_model.py
│   ├── region_sales_potential_model.py
│   ├── sales_forecast_model.py
│   └── association_rules_model.py
│
├── models/                         # 训练好的模型文件 (70+)
│   ├── refund_xgboost.joblib
│   ├── price_elasticity_best.joblib
│   ├── sales_forecast_xgboost_optimized.joblib
│   ├── product_hotness_best.joblib
│   ├── region_sales_potential_model.joblib
│   ├── region_inventory_best.joblib
│   ├── payment_conversion_best.joblib
│   ├── product_clustering_model.joblib
│   └── ...
│
├── model_metrics/                  # 模型评估指标和预测结果
│   ├── refund_metrics.json
│   ├── region_potential_prediction.csv
│   ├── blank_market_suggestions.csv
│   ├── bundle_suggestions.json
│   └── product_clusters.json
│
├── model_charts/                   # 模型可视化图表
│   ├── product_clustering.png
│   ├── co_occurrence_heatmap.png
│   └── ...
│
├── cleaned_data/                   # 清洗后的数据
├── china_provinces_geo.json        # 中国省份 GeoJSON
├── 01数据分析资料.pdf               # 参考文档
│
├── tmall_order_report.xlsx         # 天猫订单数据
├── 双十一淘宝美妆数据.xlsx          # 双十一美妆数据
└── 日化.xlsx                       # 日化销售数据
```

---

## ⚠️ 注意事项

1. **模型兼容性**：部分 GradientBoosting 模型因 numpy 版本差异无法加载，已使用 XGBoost 替代版本
2. **大文件**：`sales_forecast_randomforest_optimized.joblib` (54MB) 超过 GitHub 建议的 50MB，但可正常使用
3. **数据隐私**：Excel 数据文件包含敏感商业信息，请勿公开分享
4. **PyTorch 模型**：`.pt` 文件需 `torch` 库加载，目前使用 joblib 版本替代
