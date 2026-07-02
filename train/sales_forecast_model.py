# -*- coding: utf-8 -*-
"""
销量预测模型（优化版）

优化内容：
1. 异常值处理：过滤销量极端值
2. Log变换：处理销量长尾分布
3. 特征增强：添加周期性、促销、季节性特征
4. 模型优化：调参、修复LSTM预测逻辑
5. 多目标预测：支持7天/14天/30天销量预测
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
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
except ImportError:
    print("请安装 sklearn: pip install scikit-learn")

# PyTorch LSTM
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# 配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
METRICS_DIR = os.path.join(BASE_DIR, 'model_metrics')
CHARTS_DIR = os.path.join(BASE_DIR, 'model_charts')

for dir_path in [MODEL_DIR, METRICS_DIR, CHARTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================
# PyTorch LSTM 模型（修复版）
# ============================================================

class SalesLSTM(nn.Module):
    """销量预测LSTM模型"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(SalesLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out.squeeze()


class LSTMRegressor:
    """LSTM回归器包装类"""
    def __init__(self, input_dim, seq_len=7, hidden_dim=128, num_layers=2,
                 learning_rate=0.001, epochs=100, batch_size=64, device=None):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else DEVICE
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def create_sequences(self, X, y):
        """创建时序样本"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        sequences = []
        targets = []
        
        # 为每个样本创建序列窗口
        for i in range(len(X_scaled)):
            if i >= self.seq_len:
                # 使用过去 seq_len 个样本的特征
                seq = X_scaled[i-self.seq_len:i]
                sequences.append(seq)
                targets.append(y_scaled[i])
            else:
                # 前 seq_len 个样本用0填充
                seq = np.zeros((self.seq_len, self.input_dim))
                seq[-(i+1):] = X_scaled[:i+1]
                sequences.append(seq)
                targets.append(y_scaled[i])
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, X, y):
        """训练模型"""
        print(f"    设备: {self.device}")
        
        # 创建时序数据
        X_seq, y_seq = self.create_sequences(X, y)
        
        # 转为Tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        self.model = SalesLSTM(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # 训练
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(loader)
                print(f"      Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        """预测"""
        X_scaled = self.scaler_X.transform(X)
        
        # 为每个样本创建序列
        sequences = []
        for i in range(len(X_scaled)):
            if i >= self.seq_len:
                seq = X_scaled[i-self.seq_len:i]
            else:
                seq = np.zeros((self.seq_len, self.input_dim))
                seq[-(i+1):] = X_scaled[:i+1]
            sequences.append(seq)
        
        X_seq = np.array(sequences)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()
        
        # 反标准化
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return pred
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'input_dim': self.input_dim,
            'seq_len': self.seq_len,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.input_dim = checkpoint['input_dim']
        self.seq_len = checkpoint['seq_len']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.model = SalesLSTM(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ============================================================
# 数据加载与预处理
# ============================================================

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("正在加载日化销售数据...")
    
    sales_path = os.path.join(BASE_DIR, '日化.xlsx')
    df_sales = pd.read_excel(sales_path, sheet_name='销售订单表')
    df_product = pd.read_excel(sales_path, sheet_name='商品信息表')
    
    df = df_sales.merge(df_product, on='商品编号', how='left')
    
    # 数值转换
    df['订购数量'] = pd.to_numeric(df['订购数量'], errors='coerce').fillna(0)
    df['订购单价'] = pd.to_numeric(df['订购单价'], errors='coerce').fillna(0)
    df['金额'] = pd.to_numeric(df['金额'], errors='coerce').fillna(0)
    df['销售单价'] = pd.to_numeric(df['销售单价'], errors='coerce').fillna(0)
    
    # 日期处理
    df['订单日期'] = pd.to_datetime(df['订单日期'], errors='coerce')
    df = df[df['订单日期'].notna()]
    
    # 过滤异常值
    print(f"  原始记录: {len(df)}")
    
    # 过滤负数销量
    df = df[df['订购数量'] > 0]
    
    # 过滤极端销量（超过3倍标准差）
    mean_qty = df['订购数量'].mean()
    std_qty = df['订购数量'].std()
    upper_bound = mean_qty + 3 * std_qty
    df = df[df['订购数量'] <= upper_bound]
    
    print(f"  清洗后记录: {len(df)}")
    print(f"  时间范围: {df['订单日期'].min()} ~ {df['订单日期'].max()}")
    print(f"  商品数: {df['商品编号'].nunique()}")
    
    return df


# ============================================================
# 特征工程（增强版）
# ============================================================

def create_enhanced_features(df):
    """创建增强特征"""
    print("\n【特征工程 - 增强版】")
    
    # 按商品+日期聚合
    daily_sales = df.groupby(['商品编号', '订单日期']).agg({
        '订购数量': 'sum',
        '金额': 'sum',
        '客户编码': 'nunique',
        '订单编码': 'nunique',
        '商品大类': 'first',
        '商品小类': 'first',
        '销售单价': 'first',
        '所在省份': 'first',
        '所在区域': 'first'
    }).reset_index()
    
    daily_sales = daily_sales.sort_values(['商品编号', '订单日期'])
    
    ref_date = daily_sales['订单日期'].max()
    
    # ===== 时间特征 =====
    daily_sales['月份'] = daily_sales['订单日期'].dt.month
    daily_sales['季度'] = daily_sales['订单日期'].dt.quarter
    daily_sales['星期'] = daily_sales['订单日期'].dt.weekday
    daily_sales['是否周末'] = (daily_sales['星期'] >= 5).astype(int)
    daily_sales['日期'] = daily_sales['订单日期'].dt.day
    
    # 促销月标记
    daily_sales['是否促销月'] = daily_sales['月份'].apply(lambda x: 1 if x in [6, 11, 12] else 0)
    
    # 月初月末标记
    daily_sales['是否月初'] = (daily_sales['日期'] <= 5).astype(int)
    daily_sales['是否月末'] = (daily_sales['日期'] >= 25).astype(int)
    
    # 季节性因子
    season_map = {1: '冬', 2: '冬', 3: '春', 4: '春', 5: '春', 6: '夏',
                  7: '夏', 8: '夏', 9: '秋', 10: '秋', 11: '秋', 12: '冬'}
    daily_sales['季节'] = daily_sales['月份'].map(season_map)
    
    # ===== 历史销量特征 =====
    for lag in [1, 3, 7, 14, 21, 30]:
        daily_sales[f'销量_lag{lag}'] = daily_sales.groupby('商品编号')['订购数量'].shift(lag)
        daily_sales[f'销量_lag{lag}'] = daily_sales[f'销量_lag{lag}'].fillna(0)
    
    # 移动平均
    for window in [7, 14, 30]:
        daily_sales[f'销量_ma{window}'] = daily_sales.groupby('商品编号')['订购数量'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # 移动标准差（波动性）
    daily_sales['销量_std7'] = daily_sales.groupby('商品编号')['订购数量'].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    ).fillna(0)
    
    # 增长率
    daily_sales['销量环比'] = daily_sales.groupby('商品编号')['订购数量'].pct_change().fillna(0)
    daily_sales['销量环比'] = daily_sales['销量环比'].replace([np.inf, -np.inf], 0)
    
    # 与移动平均偏差
    daily_sales['销量偏差'] = (daily_sales['订购数量'] - daily_sales['销量_ma7']) / daily_sales['销量_ma7'].replace(0, 1)
    daily_sales['销量偏差'] = daily_sales['销量偏差'].replace([np.inf, -np.inf], 0)
    
    # ===== 商品累计特征 =====
    # 商品历史累计销量
    daily_sales['商品累计销量'] = daily_sales.groupby('商品编号')['订购数量'].cumsum()
    
    # 商品历史平均销量
    daily_sales['商品历史均值'] = daily_sales.groupby('商品编号')['订购数量'].transform(
        lambda x: x.expanding().mean()
    )
    
    # 商品销售天数
    daily_sales['商品销售天数'] = daily_sales.groupby('商品编号').cumcount() + 1
    
    # ===== 区域特征 =====
    # 区域历史销量均值
    region_avg = daily_sales.groupby('所在区域')['订购数量'].mean().to_dict()
    daily_sales['区域均值'] = daily_sales['所在区域'].map(region_avg).fillna(daily_sales['订购数量'].mean())
    
    # ===== 编码 =====
    le_category = LabelEncoder()
    daily_sales['大类编码'] = le_category.fit_transform(daily_sales['商品大类'].astype(str))
    
    le_subcategory = LabelEncoder()
    daily_sales['小类编码'] = le_subcategory.fit_transform(daily_sales['商品小类'].astype(str))
    
    le_province = LabelEncoder()
    daily_sales['省份编码'] = le_province.fit_transform(daily_sales['所在省份'].astype(str))
    
    le_season = LabelEncoder()
    daily_sales['季节编码'] = le_season.fit_transform(daily_sales['季节'].astype(str))
    
    le_region = LabelEncoder()
    daily_sales['区域编码'] = le_region.fit_transform(daily_sales['所在区域'].astype(str))
    
    print(f"  特征数量: {len(daily_sales.columns)}")
    print(f"  样本数: {len(daily_sales)}")
    
    return daily_sales, le_category, le_subcategory, le_province, le_season, le_region


# ============================================================
# 目标处理
# ============================================================

def prepare_targets(daily_sales):
    """准备多目标预测"""
    print("\n【目标准备】")
    
    # 当前销量
    daily_sales['目标_当天'] = daily_sales['订购数量']
    
    # 未来7天销量（滚动求和）
    daily_sales['目标_7天'] = daily_sales.groupby('商品编号')['订购数量'].transform(
        lambda x: x.rolling(7, min_periods=1).sum().shift(-7)
    )
    
    # 未来14天销量
    daily_sales['目标_14天'] = daily_sales.groupby('商品编号')['订购数量'].transform(
        lambda x: x.rolling(14, min_periods=1).sum().shift(-14)
    )
    
    # 未来30天销量
    daily_sales['目标_30天'] = daily_sales.groupby('商品编号')['订购数量'].transform(
        lambda x: x.rolling(30, min_periods=1).sum().shift(-30)
    )
    
    # Log变换处理长尾分布
    daily_sales['目标_当天_log'] = np.log1p(daily_sales['目标_当天'])
    
    print(f"  目标变量准备完成")
    
    return daily_sales


# ============================================================
# 模型训练（优化版）
# ============================================================

def train_optimized_models(X_train, y_train, X_test, y_test, use_log=False):
    """训练优化模型"""
    results = {}
    
    print("\n【模型训练 - 优化版】")
    
    # 如果使用log变换
    if use_log:
        y_train_model = np.log1p(y_train)
        y_test_model = np.log1p(y_test)
    else:
        y_train_model = y_train
        y_test_model = y_test
    
    # ===== XGBoost（调参） =====
    print("\n  训练 XGBoost（调参版）...")
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train_model)
    y_pred_log = xgb_model.predict(X_test)
    
    if use_log:
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)  # 确保非负
    else:
        y_pred = y_pred_log
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    
    results['XGBoost'] = {'model': xgb_model, 'metrics': metrics, 'y_pred': y_pred}
    print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.1f}%")
    
    # ===== LightGBM（调参） =====
    if HAS_LIGHTGBM:
        print("\n  训练 LightGBM（调参版）...")
        lgb_model = LGBMRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.03,
            num_leaves=50,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train_model)
        y_pred_log = lgb_model.predict(X_test)
        
        if use_log:
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)
        else:
            y_pred = y_pred_log
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        results['LightGBM'] = {'model': lgb_model, 'metrics': metrics, 'y_pred': y_pred}
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.1f}%")
    
    # ===== RandomForest（调参） =====
    print("\n  训练 RandomForest（调参版）...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train_model)
    y_pred_log = rf_model.predict(X_test)
    
    if use_log:
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)
    else:
        y_pred = y_pred_log
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    
    results['RandomForest'] = {'model': rf_model, 'metrics': metrics, 'y_pred': y_pred}
    print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.1f}%")
    
    # ===== GradientBoosting（调参） =====
    print("\n  训练 GradientBoosting（调参版）...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    gb_model.fit(X_train, y_train_model)
    y_pred_log = gb_model.predict(X_test)
    
    if use_log:
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)
    else:
        y_pred = y_pred_log
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    
    results['GradientBoosting'] = {'model': gb_model, 'metrics': metrics, 'y_pred': y_pred}
    print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.1f}%")
    
    # ===== LSTM（修复版） =====
    if HAS_TORCH and len(X_train) > 500:
        print("\n  训练 LSTM（修复版）...")
        
        lstm_model = LSTMRegressor(
            input_dim=X_train.shape[1],
            seq_len=7,
            hidden_dim=128,
            num_layers=2,
            learning_rate=0.001,
            epochs=80,
            batch_size=64,
            device=DEVICE
        )
        
        # LSTM不使用log变换，直接训练原始值
        lstm_model.fit(X_train, y_train)
        y_pred = lstm_model.predict(X_test)
        
        # 确保预测非负
        y_pred = np.maximum(y_pred, 0)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        results['LSTM'] = {'model': lstm_model, 'metrics': metrics, 'y_pred': y_pred, 'is_pytorch': True}
        print(f"    RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.1f}%")
    
    return results


# ============================================================
# 可视化
# ============================================================

def plot_results(results, y_test, charts_dir):
    """可视化"""
    
    # R²对比
    plt.figure(figsize=(10, 5))
    models = list(results.keys())
    r2_values = [results[m]['metrics']['r2'] for m in models]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    bars = plt.bar(models, r2_values, color=colors[:len(models)])
    plt.ylabel('R²')
    plt.title('销量预测模型 R² 对比（优化版）')
    plt.ylim(0, 1)
    
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'sales_forecast_r2_optimized.png'), dpi=150)
    plt.close()
    
    # 预测vs实际
    best_model = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
    y_pred = results[best_model]['y_pred']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='理想预测', linewidth=2)
    
    plt.xlabel('实际销量')
    plt.ylabel('预测销量')
    plt.title(f'最佳模型 {best_model}: 预测 vs 实际')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'sales_forecast_scatter_optimized.png'), dpi=150)
    plt.close()
    
    # MAPE对比
    plt.figure(figsize=(10, 5))
    mape_values = [results[m]['metrics']['mape'] for m in models]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    bars = plt.bar(models, mape_values, color=colors[:len(models)])
    plt.ylabel('MAPE (%)')
    plt.title('销量预测模型 MAPE 对比')
    
    for i, v in enumerate(mape_values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'sales_forecast_mape.png'), dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("销量预测模型（优化版）")
    print("=" * 60)
    
    # 1. 数据加载与预处理
    df = load_and_preprocess_data()
    
    # 2. 特征工程
    daily_sales, le_category, le_subcategory, le_province, le_season, le_region = create_enhanced_features(df)
    
    # 3. 目标准备
    daily_sales = prepare_targets(daily_sales)
    
    # 4. 构建特征矩阵
    features = [
        # 时间特征
        '月份', '季度', '星期', '是否周末', '日期', '是否促销月', '是否月初', '是否月末', '季节编码',
        # 商品特征
        '大类编码', '小类编码', '省份编码', '区域编码', '销售单价',
        # 历史销量
        '销量_lag1', '销量_lag3', '销量_lag7', '销量_lag14', '销量_lag21', '销量_lag30',
        '销量_ma7', '销量_ma14', '销量_ma30', '销量_std7',
        # 增长与偏差
        '销量环比', '销量偏差',
        # 累计特征
        '商品累计销量', '商品历史均值', '商品销售天数',
        # 区域特征
        '区域均值'
    ]
    
    X = daily_sales[features].fillna(0).values
    y = daily_sales['目标_当天'].values
    
    # 过滤有效样本（目标不为空）
    valid_mask = y > 0
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"\n数据统计:")
    print(f"  有效样本数: {len(X)}")
    print(f"  销量范围: {y.min()} ~ {y.max()}")
    print(f"  销量均值: {y.mean():.1f}")
    print(f"  销量中位数: {np.median(y):.1f}")
    
    # 5. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  测试集: {len(X_test)}")
    
    # 6. 模型训练（使用log变换）
    results = train_optimized_models(X_train, y_train, X_test, y_test, use_log=True)
    
    # 7. 模型对比
    print("\n【模型对比】")
    
    metrics_table = []
    for name, result in results.items():
        metrics_table.append({
            '模型': name,
            'RMSE': f"{result['metrics']['rmse']:.2f}",
            'MAE': f"{result['metrics']['mae']:.2f}",
            'R²': f"{result['metrics']['r2']:.4f}",
            'MAPE': f"{result['metrics']['mape']:.1f}%"
        })
    
    metrics_df = pd.DataFrame(metrics_table)
    metrics_df_sorted = metrics_df.sort_values('R²', key=lambda x: x.str.replace('%', '').astype(float), ascending=False)
    print("\n" + metrics_df_sorted.to_string(index=False))
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
    best_result = results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name} (R²={best_result['metrics']['r2']:.4f})")
    
    # 8. 保存模型
    print("\n【保存模型】")
    
    for name, result in results.items():
        if result.get('is_pytorch', False):
            # PyTorch模型使用.pt保存
            result['model'].save(os.path.join(MODEL_DIR, f'sales_forecast_{name.lower()}_optimized.pt'))
        else:
            # 其他模型使用joblib保存
            joblib.dump(result['model'], os.path.join(MODEL_DIR, f'sales_forecast_{name.lower()}_optimized.joblib'))
    
    # 保存最佳模型
    if best_result.get('is_pytorch', False):
        best_result['model'].save(os.path.join(MODEL_DIR, 'sales_forecast_best_optimized.pt'))
    else:
        joblib.dump(best_result['model'], os.path.join(MODEL_DIR, 'sales_forecast_best_optimized.joblib'))
    
    # 保存编码器
    encoders = {
        'category_encoder': le_category,
        'subcategory_encoder': le_subcategory,
        'province_encoder': le_province,
        'season_encoder': le_season,
        'region_encoder': le_region,
        'features': features,
        'use_log': True
    }
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'sales_forecast_encoders_optimized.joblib'))
    
    # 保存指标
    metrics_dict = {name: result['metrics'] for name, result in results.items()}
    with open(os.path.join(METRICS_DIR, 'sales_forecast_metrics_optimized.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)
    
    metrics_df.to_csv(os.path.join(METRICS_DIR, 'sales_forecast_metrics_optimized.csv'), index=False, encoding='utf-8-sig')
    
    # 9. 可视化
    print("\n【可视化】")
    plot_results(results, y_test, CHARTS_DIR)
    
    # 10. 特征重要性
    print("\n【特征重要性】")
    for name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
        if name in results and hasattr(results[name]['model'], 'feature_importances_'):
            importance = pd.DataFrame({
                '特征': features,
                '重要性': results[name]['model'].feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print(f"\n{name} TOP10:")
            for _, row in importance.head(10).iterrows():
                print(f"  {row['特征']}: {row['重要性']:.4f}")
    
    # 11. 样例预测
    print("\n【样例预测】")
    sample_idx = np.random.choice(len(X_test), 5, replace=False)
    for idx in sample_idx:
        actual = y_test[idx]
        pred = best_result['y_pred'][idx]
        error_pct = abs(pred - actual) / actual * 100 if actual > 0 else 0
        print(f"  实际: {actual:.0f}, 预测: {pred:.0f}, 误差: {error_pct:.1f}%")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n优化效果:")
    print(f"  最佳 R²: {best_result['metrics']['r2']:.4f}")
    print(f"  最佳 MAPE: {best_result['metrics']['mape']:.1f}%")
    
    return results, encoders


if __name__ == "__main__":
    results, encoders = main()