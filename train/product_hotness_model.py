# -*- coding: utf-8 -*-
"""
商品热度分级模型

预测目标(Y): 商品热度等级（热销/正常/滞销预警/呆滞）
业务场景: 热度分级 → 仓库存储策略差异化
- 热销商品：优先备货、多仓库分布、安全库存+50%
- 正常商品：常规备货、标准库存
- 滞销预警：减少备货、促销清仓
- 呆滞商品：建议下架、仓库释放

输入特征(X):
- 销量特征: 近30天销量、近60天销量、销量趋势
- 金额特征: 近30天销售额、客单价
- 品类特征: 商品大类、商品小类
- 时间特征: 上次销售距今天数
- 区域特征: 覆盖区域数

模型列表:
- 分类模型: XGBoost, RandomForest, LogisticRegression
- 聚类模型: K-Means (自动发现销量模式)
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
    from xgboost import XGBClassifier
except ImportError:
    print("请安装 xgboost: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
except ImportError:
    print("请安装 sklearn: pip install scikit-learn")

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
# 热度分级标准
# ============================================================

HOTNESS_RULES = {
    '热销': {
        'condition': '近30天销量 > 平均×2',
        'strategy': '优先备货、多仓库分布、安全库存+50%'
    },
    '正常': {
        'condition': '近30天销量在平均±50%',
        'strategy': '常规备货、标准库存'
    },
    '滞销预警': {
        'condition': '近30天销量 < 平均×0.5',
        'strategy': '减少备货、促销清仓'
    },
    '呆滞': {
        'condition': '近60天无销量',
        'strategy': '建议下架、仓库释放'
    }
}


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
    print(f"  商品数: {df['商品编号'].nunique()}")
    
    return df, df_product


# ============================================================
# 特征工程
# ============================================================

def create_product_features(df, df_product):
    """创建商品热度特征"""
    print("\n【特征工程】")
    
    # 参考日期（最新订单日期）
    ref_date = df['订单日期'].max()
    
    # 按商品聚合
    product_stats = df.groupby('商品编号').agg({
        '订购数量': ['sum', 'mean', 'count'],
        '金额': ['sum', 'mean'],
        '订单日期': ['min', 'max'],
        '所在省份': 'nunique',
        '客户编码': 'nunique'
    }).reset_index()
    
    product_stats.columns = ['商品编号', '总销量', '平均销量', '订单次数',
                             '总销售额', '平均销售额', '首次销售', '最后销售',
                             '覆盖省份', '客户数']
    
    # 近30天销量
    recent_30 = df[df['订单日期'] >= ref_date - pd.Timedelta(days=30)]
    recent_30_stats = recent_30.groupby('商品编号')['订购数量'].sum().reset_index()
    recent_30_stats.columns = ['商品编号', '近30天销量']
    
    # 近60天销量
    recent_60 = df[df['订单日期'] >= ref_date - pd.Timedelta(days=60)]
    recent_60_stats = recent_60.groupby('商品编号')['订购数量'].sum().reset_index()
    recent_60_stats.columns = ['商品编号', '近60天销量']
    
    # 近90天销量
    recent_90 = df[df['订单日期'] >= ref_date - pd.Timedelta(days=90)]
    recent_90_stats = recent_90.groupby('商品编号')['订购数量'].sum().reset_index()
    recent_90_stats.columns = ['商品编号', '近90天销量']
    
    # 合合特征
    product_stats = product_stats.merge(recent_30_stats, on='商品编号', how='left')
    product_stats = product_stats.merge(recent_60_stats, on='商品编号', how='left')
    product_stats = product_stats.merge(recent_90_stats, on='商品编号', how='left')
    product_stats = product_stats.merge(df_product[['商品编号', '商品大类', '商品小类', '销售单价']], 
                                        on='商品编号', how='left')
    
    # 填充缺失值
    product_stats['近30天销量'] = product_stats['近30天销量'].fillna(0)
    product_stats['近60天销量'] = product_stats['近60天销量'].fillna(0)
    product_stats['近90天销量'] = product_stats['近90天销量'].fillna(0)
    
    # 最后销售距今天数
    product_stats['距今天数'] = (ref_date - product_stats['最后销售']).dt.days
    
    # 销量趋势（近30天/近90天比值）
    product_stats['销量趋势'] = product_stats['近30天销量'] / product_stats['近90天销量'].replace(0, 1)
    product_stats['销量趋势'] = product_stats['销量趋势'].replace([np.inf, -np.inf], 0)
    
    # 热度标签（基于销量分布四分位数，更合理）
    # 使用总销量和订单次数综合判断
    total_sales_q75 = product_stats['总销量'].quantile(0.75)
    total_sales_q50 = product_stats['总销量'].quantile(0.50)
    total_sales_q25 = product_stats['总销量'].quantile(0.25)
    
    order_count_q75 = product_stats['订单次数'].quantile(0.75)
    order_count_q50 = product_stats['订单次数'].quantile(0.50)
    
    def assign_hotness(row):
        # 综合判断：总销量 + 订单次数
        score = 0
        
        # 总销量评分
        if row['总销量'] >= total_sales_q75:
            score += 2
        elif row['总销量'] >= total_sales_q50:
            score += 1
        elif row['总销量'] >= total_sales_q25:
            score += 0
        else:
            score -= 1
        
        # 订单次数评分
        if row['订单次数'] >= order_count_q75:
            score += 2
        elif row['订单次数'] >= order_count_q50:
            score += 1
        
        # 根据综合得分判断
        if score >= 3:
            return '热销'
        elif score >= 1:
            return '正常'
        elif score >= -1:
            return '滞销预警'
        else:
            return '呆滞'
    
    product_stats['热度标签'] = product_stats.apply(assign_hotness, axis=1)
    
    # 编码
    le_category = LabelEncoder()
    product_stats['大类编码'] = le_category.fit_transform(product_stats['商品大类'].astype(str))
    
    le_subcategory = LabelEncoder()
    product_stats['小类编码'] = le_subcategory.fit_transform(product_stats['商品小类'].astype(str))
    
    le_hotness = LabelEncoder()
    product_stats['热度编码'] = le_hotness.fit_transform(product_stats['热度标签'])
    
    print(f"  商品数: {len(product_stats)}")
    print(f"  热度分布: {product_stats['热度标签'].value_counts().to_dict()}")
    
    return product_stats, le_category, le_subcategory, le_hotness


# ============================================================
# 模型训练
# ============================================================

def train_classification_models(X_train, y_train, X_test, y_test):
    """训练分类模型"""
    results = {}
    
    print("\n【分类模型训练】")
    
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, use_label_encoder=False, 
                                 eval_metric='mlogloss', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\n  训练 {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {'model': model, 'metrics': metrics, 'confusion_matrix': cm, 'y_pred': y_pred}
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    return results


def train_kmeans_clustering(X, product_stats):
    """K-Means聚类模型"""
    print("\n【聚类模型 - K-Means】")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 聚类
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 分析聚类结果
    product_stats['聚类标签'] = clusters
    
    # 聚类统计
    cluster_stats = product_stats.groupby('聚类标签').agg({
        '近30天销量': 'mean',
        '总销量': 'mean',
        '总销售额': 'mean'
    }).reset_index()
    
    print("\n  聚类统计:")
    print(cluster_stats.to_string(index=False))
    
    return kmeans, scaler, clusters


# ============================================================
# 可视化
# ============================================================

def plot_results(results, y_test, product_stats, charts_dir):
    """可视化结果"""
    
    # 热度分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 饼图
    hotness_counts = product_stats['热度标签'].value_counts()
    colors = {'热销': '#2ecc71', '正常': '#3498db', '滞销预警': '#f39c12', '呆滞': '#e74c3c'}
    axes[0].pie(hotness_counts.values, labels=hotness_counts.index, 
                colors=[colors.get(l, '#999') for l in hotness_counts.index],
                autopct='%1.1f%%')
    axes[0].set_title('商品热度分布')
    
    # 柱状图
    axes[1].bar(hotness_counts.index, hotness_counts.values, 
                color=[colors.get(l, '#999') for l in hotness_counts.index])
    axes[1].set_title('商品热度数量')
    axes[1].set_ylabel('商品数')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'product_hotness_distribution.png'), dpi=150)
    plt.close()
    
    # 模型准确率对比
    plt.figure(figsize=(8, 4))
    acc_values = [results[name]['metrics']['accuracy'] for name in results.keys()]
    plt.bar(results.keys(), acc_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    plt.ylabel('Accuracy')
    plt.title('商品热度分类模型准确率')
    plt.ylim(0, 1)
    for i, v in enumerate(acc_values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'product_hotness_accuracy.png'), dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("商品热度分级模型")
    print("=" * 60)
    
    # 1. 加载数据
    df, df_product = load_sales_data()
    
    # 2. 特征工程
    product_stats, le_category, le_subcategory, le_hotness = create_product_features(df, df_product)
    
    # 3. 构建特征矩阵
    features = ['总销量', '平均销量', '订单次数', '总销售额', '平均销售额',
                '覆盖省份', '客户数', '近30天销量', '近60天销量', '近90天销量',
                '距今天数', '销量趋势', '大类编码', '小类编码']
    
    X = product_stats[features].fillna(0).values
    y = product_stats['热度编码'].values
    
    print(f"\n数据统计:")
    print(f"  商品数: {len(X)}")
    
    # 4. 数据划分（移除stratify，因为商品数量较少）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)}")
    print(f"  测试集: {len(X_test)}")
    
    # 5. 分类模型训练
    results = train_classification_models(X_train, y_train, X_test, y_test)
    
    # 6. K-Means聚类
    kmeans, scaler, clusters = train_kmeans_clustering(X, product_stats)
    
    # 7. 模型对比
    print("\n【模型对比】")
    
    metrics_table = []
    for name, result in results.items():
        metrics_table.append({
            '模型': name,
            'Accuracy': result['metrics']['accuracy'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'F1': result['metrics']['f1']
        })
    
    metrics_df = pd.DataFrame(metrics_table).sort_values('Accuracy', ascending=False)
    print("\n" + metrics_df.to_string(index=False))
    
    best_model_name = metrics_df.iloc[0]['模型']
    best_result = results[best_model_name]
    
    print(f"\n最佳模型: {best_model_name} (Accuracy={best_result['metrics']['accuracy']:.4f})")
    
    # 8. 保存模型
    print("\n【保存模型】")
    
    for name, result in results.items():
        joblib.dump(result['model'], os.path.join(MODEL_DIR, f'product_hotness_{name.lower()}.joblib'))
    
    joblib.dump(best_result['model'], os.path.join(MODEL_DIR, 'product_hotness_best.joblib'))
    joblib.dump(kmeans, os.path.join(MODEL_DIR, 'product_hotness_kmeans.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'product_hotness_scaler.joblib'))
    
    # 保存编码器
    encoders = {
        'category_encoder': le_category,
        'subcategory_encoder': le_subcategory,
        'hotness_encoder': le_hotness,
        'features': features,
        'hotness_rules': HOTNESS_RULES
    }
    joblib.dump(encoders, os.path.join(MODEL_DIR, 'product_hotness_encoders.joblib'))
    
    # 保存热度分析结果
    product_stats.to_csv(os.path.join(METRICS_DIR, 'product_hotness_analysis.csv'), 
                         index=False, encoding='utf-8-sig')
    
    # 保存指标
    with open(os.path.join(METRICS_DIR, 'product_hotness_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({name: result['metrics'] for name, result in results.items()}, f, indent=2)
    
    metrics_df.to_csv(os.path.join(METRICS_DIR, 'product_hotness_metrics.csv'), index=False, encoding='utf-8-sig')
    
    # 9. 可视化
    print("\n【可视化】")
    plot_results(results, y_test, product_stats, CHARTS_DIR)
    
    # 10. 热度策略输出
    print("\n【热度分级策略】")
    for hotness, rule in HOTNESS_RULES.items():
        count = len(product_stats[product_stats['热度标签'] == hotness])
        print(f"\n  {hotness} ({count}个商品):")
        print(f"    条件: {rule['condition']}")
        print(f"    策略: {rule['strategy']}")
    
    # 11. 特征重要性
    if hasattr(best_result['model'], 'feature_importances_'):
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
    
    return results, encoders, product_stats


if __name__ == "__main__":
    results, encoders, product_stats = main()