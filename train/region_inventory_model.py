# -*- coding: utf-8 -*-
"""
区域库存需求预测模型

预测目标(Y): 各省份/区域未来30天销量
业务场景: 预测区域销量 → 计算各仓库库存需求 → 动态调拨策略

输入特征(X):
- 区域销量特征: 近30天销量、近60天销量、销量趋势
- 区域客户特征: 客户数、客户活跃度、新客户占比
- 时间特征: 月份、季度、促销月标记
- 商品特征: 品类分布（各品类销量占比）
- 区域渗透率: 已覆盖地市数/总地市数

模型列表:
- 回归模型: XGBoost, LightGBM, RandomForest
- 深度模型: PyTorch MLP (支持GPU)

输出应用:
- 热销区域多备货、滞销区域减少库存
- 计算安全库存、补货点
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except ImportError:
    print("请安装 xgboost: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    print("请安装 sklearn: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch 设备: {DEVICE}")
except ImportError:
    print("未安装 torch: pip install torch")
    HAS_TORCH = False
    DEVICE = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("请安装 matplotlib: pip install matplotlib")


# ============================================================
# 配置路径
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'model_metrics')
CHARTS_DIR = os.path.join(BASE_DIR, 'model_charts')

for dir_path in [MODEL_DIR, METRICS_DIR, CHARTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================
# PyTorch MLP 模型
# ============================================================

class RegionMLP(nn.Module):
    """区域库存预测MLP模型"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(RegionMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PyTorchRegressor:
    """PyTorch回归器包装"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], learning_rate=0.001,
                 epochs=100, batch_size=32, device=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else (DEVICE if HAS_TORCH else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = RegionMLP(self.input_dim, self.hidden_dims).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy().flatten()
        
        return pred
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.scaler = checkpoint['scaler']
        self.model = RegionMLP(checkpoint['input_dim'], checkpoint['hidden_dims']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ============================================================
# 数据加载
# ============================================================

def load_sales_data():
    """加载日化销售数据"""
    print("正在加载日化销售数据...")
    
    sales_path = os.path.join(BASE_DIR, '日化.xlsx')
    df_sales = pd.read_excel(sales_path, sheet_name='销售订单表')
    df_product = pd.read_excel(sales_path, sheet_name='商品信息表')
    
    df = df_sales.merge(df_product, on='商品编号', how='left')
    
    df['订购数量'] = pd.to_numeric(df['订购数量'], errors='coerce').fillna(0)
    df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
    df['订单日期'] = pd.to_datetime(df['订单日期'], errors='coerce')
    
    df = df[df['订单日期'].notna()]
    
    print(f"  加载完成: {len(df)} 条记录")
    print(f"  省份数: {df['所在省份'].nunique()}")
    
    return df


# ============================================================
# 特征工程
# ============================================================

def create_region_features(df):
    """创建区域库存预测特征（使用省份+月份聚合，增加数据量）"""
    print("\n【特征工程】")
    
    ref_date = df['订单日期'].max()
    
    # ===== 使用省份+月份聚合（增加数据量） =====
    monthly_region = df.groupby(['所在省份', df['订单日期'].dt.to_period('M').astype(str)]).agg({
        '订购数量': 'sum',
        '金额': 'sum',
        '客户编码': 'nunique',
        '订单编码': 'nunique',
        '商品编号': 'nunique'
    }).reset_index()
    monthly_region.columns = ['省份', '月份', '销量', '销售额', '客户数', '订单数', '商品数']
    
    # 解析月份
    monthly_region['月份日期'] = pd.to_datetime(monthly_region['月份'])
    monthly_region['月'] = monthly_region['月份日期'].dt.month
    monthly_region['年'] = monthly_region['月份日期'].dt.year
    monthly_region['季度'] = monthly_region['月份日期'].dt.quarter
    
    # 促销月标记
    monthly_region['是否促销月'] = monthly_region['月'].apply(lambda x: 1 if x in [6, 11, 12] else 0)
    
    # 省份整体统计（作为特征）
    province_stats = df.groupby('所在省份').agg({
        '订购数量': ['sum', 'mean'],
        '金额': 'sum',
        '客户编码': 'nunique',
        '商品编号': 'nunique'
    }).reset_index()
    province_stats.columns = ['省份', '省份总销量', '省份平均销量', '省份总销售额', '省份客户数', '省份商品数']
    
    # 合并省份统计
    monthly_region = monthly_region.merge(province_stats, on='省份', how='left')
    
    # 计算省份销量占比
    monthly_region['省份销量占比'] = monthly_region['销量'] / monthly_region['省份总销量'].replace(0, 1)
    
    # 滞后特征（上月销量）
    monthly_region = monthly_region.sort_values(['省份', '月份日期'])
    monthly_region['上月销量'] = monthly_region.groupby('省份')['销量'].shift(1).fillna(0)
    monthly_region['上月销售额'] = monthly_region.groupby('省份')['销售额'].shift(1).fillna(0)
    
    # 销量增长率
    monthly_region['销量环比'] = monthly_region.groupby('省份')['销量'].pct_change().fillna(0)
    monthly_region['销量环比'] = monthly_region['销量环比'].replace([np.inf, -np.inf], 0)
    
    # 省份编码
    le_province = LabelEncoder()
    monthly_region['省份编码'] = le_province.fit_transform(monthly_region['省份'].astype(str))
    
    print(f"  样本数（省份×月份）: {len(monthly_region)}")
    print(f"  省份数: {monthly_region['省份'].nunique()}")
    
    return monthly_region, le_province


# ============================================================
# 模型训练
# ============================================================

def train_models(X_train, y_train, X_test, y_test):
    """训练多个回归模型"""
    results = {}
    
    print("\n【模型训练】")
    
    # 基础模型
    basic_models = {
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    for name, model in basic_models.items():
        print(f"\n  训练 {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        results[name] = {'model': model, 'metrics': metrics, 'y_pred': y_pred, 'is_pytorch': False}
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
    
    # PyTorch MLP
    if HAS_TORCH:
        print("\n【深度模型 - MLP】")
        print(f"  设备: {DEVICE}")
        
        mlp = PyTorchRegressor(
            input_dim=X_train.shape[1],
            hidden_dims=[128, 64, 32],
            epochs=50,
            device=DEVICE
        )
        
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        results['MLP'] = {'model': mlp, 'metrics': metrics, 'y_pred': y_pred, 'is_pytorch': True}
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
    
    return results


# ============================================================
# 库存参数计算
# ============================================================

def calculate_inventory_params(region_stats, predicted_sales):
    """计算库存参数"""
    print("\n【库存参数计算】")
    
    # 安全库存 = 预测销量 × 安全系数（通常1.5-2）
    safety_factor = 1.5
    region_stats['预测销量'] = predicted_sales
    region_stats['安全库存'] = region_stats['预测销量'] * safety_factor
    
    # 补货点 = 安全库存 + 预测周期销量
    lead_time_days = 7  # 补货周期7天
    region_stats['补货点'] = region_stats['安全库存'] + (region_stats['预测销量'] / 30 * lead_time_days)
    
    # 库存建议
    avg_predicted = region_stats['预测销量'].mean()
    
    def inventory_strategy(row):
        if row['预测销量'] > avg_predicted * 1.5:
            return '热销区域: 安全库存+50%、优先补货'
        elif row['预测销量'] < avg_predicted * 0.5:
            return '滞销区域: 减少库存、延迟补货'
        else:
            return '正常区域: 标准库存、常规补货'
    
    region_stats['库存策略'] = region_stats.apply(inventory_strategy, axis=1)
    
    print(f"\n  平均预测销量: {avg_predicted:.0f}")
    print(f"\n库存策略分布:")
    print(region_stats['库存策略'].value_counts().to_string())
    
    return region_stats


# ============================================================
# 可视化
# ============================================================

def plot_results(results, y_test, region_stats, charts_dir):
    """可视化结果"""
    
    # 区域销量预测对比
    plt.figure(figsize=(12, 6))
    region_stats_sorted = region_stats.sort_values('预测销量', ascending=False).head(15)
    
    plt.barh(region_stats_sorted['省份'], region_stats_sorted['预测销量'], color='#3498db')
    plt.xlabel('预测销量')
    plt.ylabel('省份')
    plt.title('区域库存需求预测 TOP15')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'region_inventory_forecast.png'), dpi=150)
    plt.close()
    
    # 模型R²对比
    plt.figure(figsize=(8, 4))
    r2_values = [results[name]['metrics']['r2'] for name in results.keys()]
    plt.bar(results.keys(), r2_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
    plt.ylabel('R²')
    plt.title('区域库存预测模型 R² 对比')
    plt.ylim(0, 1)
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'region_inventory_r2.png'), dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("区域库存需求预测模型")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_sales_data()
    
    # 2. 特征工程（省份×月份聚合）
    monthly_region, le_province = create_region_features(df)
    
    # 3. 构建特征矩阵
    features = ['月', '年', '季度', '是否促销月', '省份编码',
                '客户数', '订单数', '商品数',
                '省份总销量', '省份平均销量', '省份总销售额', '省份客户数', '省份商品数',
                '上月销量', '上月销售额', '销量环比']
    
    X = monthly_region[features].fillna(0).values
    y = monthly_region['销量'].values  # 预测目标：当月销量
    
    print(f"\n数据统计:")
    print(f"  样本数（省份×月份）: {len(X)}")
    print(f"  销量范围: {y.min()} ~ {y.max()}")
    print(f"  销量均值: {y.mean():.1f}")
    
    # 4. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  测试集: {len(X_test)}")
    
    # 5. 模型训练
    results = train_models(X_train, y_train, X_test, y_test)
    
    # 6. 模型对比
    print("\n【模型对比】")
    
    metrics_table = []
    for name, result in results.items():
        metrics_table.append({
            '模型': name,
            'RMSE': result['metrics']['rmse'],
            'MAE': result['metrics']['mae'],
            'R²': result['metrics']['r2']
        })
    
    metrics_df = pd.DataFrame(metrics_table).sort_values('R²', ascending=False)
    print("\n" + metrics_df.to_string(index=False))
    
    best_model_name = metrics_df.iloc[0]['模型']
    best_result = results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name} (R²={best_result['metrics']['r2']:.4f})")
    
    # 7. 对所有样本进行预测
    print("\n【全量预测】")
    
    if best_result['is_pytorch']:
        predicted_sales = best_result['model'].predict(X)
    else:
        predicted_sales = best_result['model'].predict(X)
    
    # 8. 计算库存参数（按省份汇总）
    monthly_region['预测销量'] = predicted_sales
    region_stats = monthly_region.groupby('省份').agg({
        '销量': 'sum',
        '预测销量': 'sum'
    }).reset_index()
    
    region_stats = calculate_inventory_params(region_stats, region_stats['预测销量'].values)
    
    # 9. 保存模型
    print("\n【保存模型】")
    
    for name, result in results.items():
        if result['is_pytorch']:
            result['model'].save(os.path.join(MODEL_DIR, f'region_inventory_{name.lower()}.pt'))
        else:
            joblib.dump(result['model'], os.path.join(MODEL_DIR, f'region_inventory_{name.lower()}.joblib'))
    
    # 保存最佳模型
    if best_result['is_pytorch']:
        best_result['model'].save(os.path.join(MODEL_DIR, 'region_inventory_best.pt'))
    else:
        joblib.dump(best_result['model'], os.path.join(MODEL_DIR, 'region_inventory_best.joblib'))
    
    # 保存编码器
    encoders = {
        'province_encoder': le_province,
        'features': features,
        'safety_factor': 1.5,
        'lead_time_days': 7
    }
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'region_inventory_encoders.joblib'))
    
    # 保存区域库存分析
    region_stats.to_csv(os.path.join(METRICS_DIR, 'region_inventory_analysis.csv'),
                        index=False, encoding='utf-8-sig')
    
    # 保存指标
    with open(os.path.join(METRICS_DIR, 'region_inventory_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({name: result['metrics'] for name, result in results.items()}, f, indent=2)
    
    metrics_df.to_csv(os.path.join(METRICS_DIR, 'region_inventory_metrics.csv'), index=False, encoding='utf-8-sig')
    
    # 10. 可视化
    print("\n【可视化】")
    plot_results(results, y_test, region_stats, CHARTS_DIR)
    
    # 11. 输出库存策略
    print("\n【库存策略建议】")
    print(region_stats[['省份', '预测销量', '安全库存', '补货点', '库存策略']].head(10).to_string(index=False))
    
    # 12. 特征重要性
    if not best_result['is_pytorch'] and hasattr(best_result['model'], 'feature_importances_'):
        importance = pd.DataFrame({
            '特征': features,
            '重要性': best_result['model'].feature_importances_
        }).sort_values('重要性', ascending=False)
        
        print("\n特征重要性 TOP5:")
        for _, row in importance.head(5).iterrows():
            print(f"  {row['特征']}: {row['重要性']:.4f}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    return results, encoders, region_stats


if __name__ == "__main__":
    results, encoders, region_stats = main()