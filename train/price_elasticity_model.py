# -*- coding: utf-8 -*-
"""
大促商品价格弹性与GMV预测模型

预测目标(Y): 给定某个价格下，该商品未来一天的预测销量（回归问题）
业务场景: 大促定价策略 - "定几折才能利润最大化？"

输入特征(X):
- 价格特征: 当前定价、价格变动幅度（相比前一天降了百分之几）
- 商品属性特征: 是否套装、所属品类
- 累计势能特征: 历史累计销量、历史累计评论数
- 大促时间节点: 预热期(11.10)、爆发期(11.11)、返场期(11.12-14)

模型列表:
- 基础模型: XGBoost, RandomForest, LinearRegression, Ridge
- 高级模型: PyTorch DNN (支持GPU), LightGBM
"""

import pandas as pd
import numpy as np
import openpyxl
import os
import joblib
import json
import warnings
import re
warnings.filterwarnings('ignore')

# 尝试导入必要的库
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except ImportError:
    print("请先安装 xgboost: pip install xgboost")

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    print("未安装 lightgbm: pip install lightgbm")
    HAS_LIGHTGBM = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch 设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("未安装 torch: pip install torch")
    HAS_TORCH = False
    DEVICE = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("请先安装 matplotlib: pip install matplotlib")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)


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
# PyTorch 深度神经网络（回归）
# ============================================================

class PriceElasticityDNN(nn.Module):
    """价格弹性预测深度神经网络"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(PriceElasticityDNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # 回归输出
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PyTorchRegressor:
    """PyTorch 回归器包装类"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3,
                 learning_rate=0.001, epochs=100, batch_size=64, device=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else (DEVICE if HAS_TORCH else 'cpu')
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fit(self, X, y):
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 转换为 Tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)
        
        # 创建 DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        self.model = PriceElasticityDNN(self.input_dim, self.hidden_dims, self.dropout_rate).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # 训练
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
                avg_loss = total_loss / len(loader)
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        # 反标准化
        return self.scaler_y.inverse_transform(predictions).flatten()
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.model = PriceElasticityDNN(
            checkpoint['input_dim'],
            checkpoint['hidden_dims'],
            checkpoint['dropout_rate']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ============================================================
# 数据加载
# ============================================================

def load_beauty_data(filepath):
    """加载双十一淘宝美妆数据"""
    print("正在加载 双十一淘宝美妆数据.xlsx...")
    df = pd.read_excel(filepath)
    
    # 标准化列名
    df.columns = ['update_time', 'id', 'title', 'price', 'sale_count', 
                  'comment_count', '店名'] if len(df.columns) == 7 else df.columns
    
    print(f"  加载完成: {len(df)} 条记录")
    print(f"  时间范围: {df['update_time'].min()} ~ {df['update_time'].max()}")
    print(f"  商品数: {df['id'].nunique()} 个")
    print(f"  店铺数: {df['店名'].nunique()} 家")
    
    return df


# ============================================================
# 特征工程
# ============================================================

def extract_product_features(title):
    """从商品标题提取特征"""
    # 是否套装
    is_suit = 0
    suit_keywords = ['套装', '组合', '礼盒', '套盒', '整套', '系列', '全套', '三件套', '两件套', '四件套']
    for kw in suit_keywords:
        if kw in title:
            is_suit = 1
            break
    
    # 品类识别
    category = '其他'
    category_map = {
        '面膜': ['面膜', '贴膜', '涂抹面膜', '泥膜'],
        '面霜': ['面霜', '霜', '日霜', '晚霜', '保湿霜'],
        '水乳': ['水乳', '爽肤水', '乳液', '精华水', '柔肤水'],
        '精华': ['精华', '精华液', '原液', '安瓶'],
        '眼霜': ['眼霜', '眼部精华', '眼膜'],
        '洁面': ['洁面', '洗面奶', '洁面乳', '洁面膏', '卸妆'],
        '防晒': ['防晒', '防晒霜', '隔离', '防晒喷雾'],
        '口红': ['口红', '唇膏', '唇彩', '唇釉'],
        '粉底': ['粉底', '粉底液', '气垫', 'BB', 'CC', '粉饼'],
        '眼影': ['眼影', '眼影盘'],
        '腮红': ['腮红', '修容', '高光'],
        '睫毛': ['睫毛', '睫毛膏', '睫毛打底'],
        '眉笔': ['眉笔', '眉粉', '染眉膏']
    }
    
    for cat, keywords in category_map.items():
        for kw in keywords:
            if kw in title:
                category = cat
                break
        if category != '其他':
            break
    
    return is_suit, category


def get_promotion_period(date):
    """判断大促时间节点"""
    if pd.isna(date):
        return '未知'
    
    date_str = str(date)
    day = date_str.split('-')[-1] if '-' in date_str else date_str.split('/')[-1]
    day = int(day.replace(' ', '').replace('日', '')) if day.isdigit() or day.replace('日', '').isdigit() else 0
    
    if day == 10:
        return '预热期'
    elif day == 11:
        return '爆发期'
    elif 12 <= day <= 14:
        return '返场期'
    else:
        return '日常'


def calculate_price_change(df):
    """计算价格变动幅度"""
    # 按商品和时间排序
    df = df.sort_values(['id', 'update_time'])
    
    # 计算每个商品前一天的价格
    df['前一天价格'] = df.groupby('id')['price'].shift(1)
    
    # 计算价格变动幅度
    df['价格变动幅度'] = (df['price'] - df['前一天价格']) / df['前一天价格']
    df['价格变动幅度'] = df['价格变动幅度'].fillna(0)
    
    # 计算降价标记
    df['是否降价'] = (df['价格变动幅度'] < 0).astype(int)
    df['降价幅度'] = df['价格变动幅度'].apply(lambda x: abs(x) if x < 0 else 0)
    
    return df


def feature_engineering(df):
    """特征工程"""
    print("\n【特征工程】")
    
    # 1. 商品属性特征
    print("  提取商品属性...")
    df['是否套装'] = df['title'].apply(lambda x: extract_product_features(x)[0])
    df['品类'] = df['title'].apply(lambda x: extract_product_features(x)[1])
    
    # 2. 大促时间节点
    print("  判断大促时间节点...")
    df['大促节点'] = df['update_time'].apply(get_promotion_period)
    
    # 3. 价格变动特征
    print("  计算价格变动...")
    df = calculate_price_change(df)
    
    # 4. 累计势能特征（历史销量和评论）
    print("  计算累计势能...")
    df['历史累计销量'] = df.groupby('id')['sale_count'].cumsum()
    df['历史累计评论'] = df.groupby('id')['comment_count'].cumsum()
    
    # 5. 店铺特征
    df['店铺销量排名'] = df.groupby('店名')['sale_count'].rank(method='dense', ascending=False)
    
    # 6. 编码
    le_category = LabelEncoder()
    df['品类编码'] = le_category.fit_transform(df['品类'])
    
    le_period = LabelEncoder()
    df['大促节点编码'] = le_period.fit_transform(df['大促节点'])
    
    le_shop = LabelEncoder()
    df['店铺编码'] = le_shop.fit_transform(df['店名'])
    
    # 7. 计算GMV
    df['gmv'] = df['price'] * df['sale_count']
    
    print(f"\n特征统计:")
    print(f"  品类分布: {df['品类'].value_counts().to_dict()}")
    print(f"  套装比例: {df['是否套装'].sum() / len(df) * 100:.1f}%")
    print(f"  大促节点分布: {df['大促节点'].value_counts().to_dict()}")
    
    return df, le_category, le_period, le_shop


# ============================================================
# 模型训练
# ============================================================

def train_models(X_train, y_train, X_test, y_test):
    """训练多个回归模型"""
    results = {}
    
    # 1. 基础模型
    print("\n【基础模型训练】")
    
    basic_models = {
        'XGBoost': XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=6, random_state=42
        )
    }
    
    for name, model in basic_models.items():
        print(f"\n  训练 {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        results[name] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'is_pytorch': False
        }
        
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
    
    # 2. LightGBM
    if HAS_LIGHTGBM:
        print("\n【高级模型训练 - LightGBM】")
        print(f"\n  训练 LightGBM...")
        
        lgb_model = LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        results['LightGBM'] = {
            'model': lgb_model,
            'metrics': metrics,
            'y_pred': y_pred,
            'is_pytorch': False
        }
        
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
    
    # 3. PyTorch DNN
    if HAS_TORCH:
        print("\n【高级模型训练 - PyTorch DNN】")
        print(f"  设备: {DEVICE}")
        print(f"\n  训练 DNN...")
        
        dnn_model = PyTorchRegressor(
            input_dim=X_train.shape[1],
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=0.3,
            learning_rate=0.001,
            epochs=100,
            batch_size=64,
            device=DEVICE
        )
        
        dnn_model.fit(X_train, y_train)
        y_pred = dnn_model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        results['DNN'] = {
            'model': dnn_model,
            'metrics': metrics,
            'y_pred': y_pred,
            'is_pytorch': True
        }
        
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}")
    
    return results


# ============================================================
# 可视化
# ============================================================

def plot_results(results, y_test, charts_dir):
    """可视化结果"""
    
    # 预测值 vs 实际值对比
    plt.figure(figsize=(12, 5))
    
    # 子图1: 各模型预测对比
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    for i, name in enumerate(model_names[:4]):  # 只显示前4个避免拥挤
        y_pred = results[name]['y_pred']
        plt.scatter(y_test, y_pred, alpha=0.5, label=name, s=20)
    
    max_val = max(y_test.max(), max([r['y_pred'].max() for r in results.values()]))
    plt.plot([0, max_val], [0, max_val], 'k--', label='理想预测')
    plt.xlabel('实际销量')
    plt.ylabel('预测销量')
    plt.title('销量预测 vs 实际值')
    plt.legend()
    
    # 子图2: 指标对比
    plt.subplot(1, 2, 2)
    metrics_names = ['rmse', 'mae', 'r2']
    x = np.arange(len(results))
    width = 0.25
    
    for i, metric in enumerate(metrics_names):
        values = [results[name]['metrics'][metric] for name in results.keys()]
        if metric == 'r2':
            # R² 单独显示，范围不同
            continue
        plt.bar(x + i * width, values, width, label=metric.upper())
    
    plt.xticks(x + width, results.keys(), rotation=45)
    plt.ylabel('误差值')
    plt.title('模型误差对比')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'price_elasticity_prediction.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # R² 对比图
    plt.figure(figsize=(10, 4))
    r2_values = [results[name]['metrics']['r2'] for name in results.keys()]
    plt.bar(results.keys(), r2_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22'])
    plt.ylabel('R²')
    plt.title('模型 R² 对比')
    plt.xticks(rotation=45)
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'price_elasticity_r2.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 价格弹性模拟
# ============================================================

def simulate_price_elasticity(model, base_features, price_range, scaler=None):
    """
    模拟不同价格下的销量预测
    用于生成价格-销量-GMV 曲线
    """
    predictions = []
    
    for price in price_range:
        # 复制基础特征
        features = base_features.copy()
        features[0] = price  # 假设第一个特征是价格
        
        # 预测销量
        if hasattr(model, 'predict'):
            pred_sales = model.predict([features])[0]
        else:
            pred_sales = model.predict(np.array([features]))[0]
        
        # 计算GMV
        gmv = price * pred_sales
        
        predictions.append({
            '价格': price,
            '预测销量': pred_sales,
            '预测GMV': gmv
        })
    
    return pd.DataFrame(predictions)


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("大促商品价格弹性与GMV预测模型")
    print("=" * 60)
    
    # 1. 加载数据
    beauty_path = os.path.join(BASE_DIR, '双十一淘宝美妆数据.xlsx')
    df = load_beauty_data(beauty_path)
    
    # 2. 特征工程
    df, le_category, le_period, le_shop = feature_engineering(df)
    
    # 3. 构建特征矩阵
    features = ['price', '价格变动幅度', '降价幅度', '是否降价', '是否套装',
                '品类编码', '历史累计销量', '历史累计评论',
                '大促节点编码', '店铺编码', '店铺销量排名']
    
    # 过滤异常值
    df_model = df[df['sale_count'] > 0].copy()
    df_model = df_model[df_model['price'] > 0].copy()
    
    X = df_model[features].fillna(0).values
    y = df_model['sale_count'].values
    
    print(f"\n数据统计:")
    print(f"  有效样本: {len(df_model)} 条")
    print(f"  销量范围: {y.min()} ~ {y.max()}")
    print(f"  价格范围: {df_model['price'].min():.1f} ~ {df_model['price'].max():.1f}")
    
    # 4. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 条")
    print(f"  测试集: {len(X_test)} 条")
    
    # 5. 模型训练
    print("\n【模型训练】")
    results = train_models(X_train, y_train, X_test, y_test)
    
    # 6. 模型对比
    print("\n【模型对比】")
    
    metrics_table = []
    for name, result in results.items():
        metrics_table.append({
            '模型': name,
            'RMSE': result['metrics']['rmse'],
            'MAE': result['metrics']['mae'],
            'R²': result['metrics']['r2'],
            'MAPE': f"{result['metrics']['mape']:.1f}%"
        })
    
    metrics_df = pd.DataFrame(metrics_table).sort_values('R²', ascending=False)
    print("\n" + metrics_df.to_string(index=False))
    
    best_model_name = metrics_df.iloc[0]['模型']
    best_result = results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name} (R²={best_result['metrics']['r2']:.4f})")
    
    # 7. 保存模型
    print("\n【保存模型】")
    
    if best_result['is_pytorch']:
        best_result['model'].save(os.path.join(MODEL_DIR, 'price_elasticity_best.pt'))
        print(f"  最佳模型: price_elasticity_best.pt (PyTorch)")
    else:
        joblib.dump(best_result['model'], os.path.join(MODEL_DIR, 'price_elasticity_best.joblib'))
        print(f"  最佳模型: price_elasticity_best.joblib")
    
    # 保存所有模型
    for name, result in results.items():
        if result['is_pytorch']:
            result['model'].save(os.path.join(MODEL_DIR, f'price_elasticity_{name.lower()}.pt'))
        else:
            joblib.dump(result['model'], os.path.join(MODEL_DIR, f'price_elasticity_{name.lower()}.joblib'))
    
    # 保存编码器
    encoders = {
        'category_encoder': le_category,
        'period_encoder': le_period,
        'shop_encoder': le_shop,
        'features': features,
        'feature_stats': {
            'price_mean': df_model['price'].mean(),
            'price_std': df_model['price'].std(),
            'sale_mean': df_model['sale_count'].mean(),
            'sale_std': df_model['sale_count'].std()
        }
    }
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'price_elasticity_encoders.joblib'))
    
    # 8. 保存指标
    metrics_dict = {name: result['metrics'] for name, result in results.items()}
    
    with open(os.path.join(METRICS_DIR, 'price_elasticity_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
    
    metrics_df.to_csv(os.path.join(METRICS_DIR, 'price_elasticity_metrics.csv'),
                      encoding='utf-8-sig', index=False)
    
    # 9. 可视化
    print("\n【可视化】")
    plot_results(results, y_test, CHARTS_DIR)
    
    # 10. 特征重要性
    tree_models = ['XGBoost', 'RandomForest', 'GradientBoosting', 'LightGBM']
    for name in tree_models:
        if name in results and hasattr(results[name]['model'], 'feature_importances_'):
            importance_df = pd.DataFrame({
                '特征': features,
                '重要性': results[name]['model'].feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print(f"\n{name} 特征重要性:")
            for _, row in importance_df.head(5).iterrows():
                print(f"  {row['特征']}: {row['重要性']:.4f}")
    
    # 11. 总结
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  模型: {MODEL_DIR}/price_elasticity_*.joblib / .pt")
    print(f"  指标: {METRICS_DIR}/price_elasticity_metrics.csv/json")
    print(f"  图表: {CHARTS_DIR}/price_elasticity_*.png")
    
    print("\n【应用场景示例】")
    print("  商家输入目标价格 → 模型预测销量 → 计算GMV")
    print("  可绘制价格-销量-GMV曲线，找到利润最优价格点")
    
    return results, encoders


if __name__ == "__main__":
    results, encoders = main()