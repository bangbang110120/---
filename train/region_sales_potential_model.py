# -*- coding: utf-8 -*-
"""
区域销售潜力预测模型
预测某商品在某地区的销售潜力，帮助商家识别空白市场机会
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 60)
    print("区域销售潜力预测模型")
    print("=" * 60)
    
    # ============================================================
    # 1. 加载数据
    # ============================================================
    data_path = os.path.join(base_dir, '日化.xlsx')
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 - {data_path}")
        return None
    
    print("正在加载日化销售数据...")
    df_sales = pd.read_excel(data_path, sheet_name='销售订单表')
    df_product = pd.read_excel(data_path, sheet_name='商品信息表')
    print(f"  销售记录: {len(df_sales)} 条")
    print(f"  商品数: {len(df_product)} 个")
    
    # ============================================================
    # 2. 数据预处理
    # ============================================================
    print("\n【数据预处理】")
    
    # 清洗数值列
    df_sales['订购数量'] = pd.to_numeric(df_sales['订购数量'], errors='coerce').fillna(0)
    df_sales['金额'] = pd.to_numeric(df_sales['金额'], errors='coerce').fillna(0)
    
    # 过滤无效数据
    df_sales = df_sales[
        (df_sales['订购数量'] > 0) & 
        (df_sales['金额'] > 0) &
        (df_sales['所在省份'].notna())
    ]
    
    # 合并商品信息
    df_merged = df_sales.merge(df_product[['商品编号', '商品大类', '商品小类', '销售单价']], 
                               on='商品编号', how='left')
    
    print(f"  有效记录: {len(df_merged)} 条")
    print(f"  省份数: {df_merged['所在省份'].nunique()}")
    print(f"  商品大类数: {df_merged['商品大类'].nunique()}")
    
    # ============================================================
    # 3. 构建区域-商品-时间聚合数据
    # ============================================================
    print("\n【构建预测数据集】")
    
    # 转换日期
    df_merged['订单日期'] = pd.to_datetime(df_merged['订单日期'], errors='coerce')
    df_merged['月份'] = df_merged['订单日期'].dt.month
    df_merged['季度'] = df_merged['订单日期'].dt.quarter
    df_merged['年份'] = df_merged['订单日期'].dt.year
    
    # 按省份-商品大类-月份聚合
    agg_data = df_merged.groupby(['所在省份', '商品大类', '月份']).agg(
        销量=('订购数量', 'sum'),
        销售额=('金额', 'sum'),
        订单数=('订单编码', 'nunique'),
        客户数=('客户编码', 'nunique'),
        平均单价=('销售单价', 'mean')
    ).reset_index()
    
    print(f"  聚合样本数: {len(agg_data)}")
    
    # ============================================================
    # 4. 特征工程
    # ============================================================
    print("\n【特征工程】")
    
    # 4a. 区域历史特征（省份维度）
    region_stats = df_merged.groupby('所在省份').agg(
        省份总销量=('订购数量', 'sum'),
        省份总销售额=('金额', 'sum'),
        省份客户数=('客户编码', 'nunique'),
        省份订单数=('订单编码', 'nunique'),
        省份覆盖品类=('商品大类', 'nunique'),
        省份覆盖商品=('商品编号', 'nunique')
    ).reset_index()
    
    # 省份活跃度得分
    region_stats['省份活跃度'] = region_stats['省份总销售额'] / region_stats['省份总销售额'].max()
    region_stats['省份渗透率'] = region_stats['省份覆盖品类'] / df_merged['商品大类'].nunique()
    
    # 4b. 品类热度特征
    category_stats = df_merged.groupby('商品大类').agg(
        品类总销量=('订购数量', 'sum'),
        品类总销售额=('金额', 'sum'),
        品类覆盖省份=('所在省份', 'nunique'),
        品类平均单价=('销售单价', 'mean')
    ).reset_index()
    
    category_stats['品类热度'] = category_stats['品类总销售额'] / category_stats['品类总销售额'].max()
    category_stats['品类渗透率'] = category_stats['品类覆盖省份'] / df_merged['所在省份'].nunique()
    
    # 合并特征
    agg_data = agg_data.merge(region_stats, on='所在省份', how='left')
    agg_data = agg_data.merge(category_stats, on='商品大类', how='left')
    
    # 4c. 交叉特征
    agg_data['省份品类匹配度'] = agg_data['省份渗透率'] * agg_data['品类渗透率']
    agg_data['价格匹配度'] = agg_data['平均单价'] / agg_data['省份总销售额'] / agg_data['省份订单数']
    agg_data['人均销量'] = agg_data['销量'] / agg_data['客户数'].replace(0, 1)
    
    # 4d. 时间特征
    agg_data['是否旺季'] = agg_data['月份'].apply(lambda x: 1 if x in [6, 7, 8, 11, 12] else 0)
    agg_data['是否促销季'] = agg_data['月份'].apply(lambda x: 1 if x in [6, 11, 12] else 0)
    
    # 编码类别特征
    le_province = LabelEncoder()
    le_category = LabelEncoder()
    
    agg_data['省份编码'] = le_province.fit_transform(agg_data['所在省份'].astype(str))
    agg_data['品类编码'] = le_category.fit_transform(agg_data['商品大类'].astype(str))
    
    # 填充缺失值
    agg_data = agg_data.fillna(0)
    
    print(f"  特征数: {len(agg_data.columns)}")
    print(f"  样本数: {len(agg_data)}")
    
    # ============================================================
    # 5. 构建模型数据
    # ============================================================
    print("\n【构建模型数据】")
    
    # 预测目标：销售额
    feature_cols = [
        '省份编码', '品类编码', '月份',
        '省份总销量', '省份总销售额', '省份客户数', '省份订单数',
        '省份覆盖品类', '省份覆盖商品', '省份活跃度', '省份渗透率',
        '品类总销量', '品类总销售额', '品类覆盖省份', '品类平均单价',
        '品类热度', '品类渗透率',
        '省份品类匹配度', '是否旺季', '是否促销季'
    ]
    
    X = agg_data[feature_cols].values
    y = agg_data['销售额'].values
    
    # Log变换目标（处理长尾分布）
    y_log = np.log1p(y)
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  训练集: {len(X_train)}")
    print(f"  测试集: {len(X_test)}")
    
    # ============================================================
    # 6. 模型训练
    # ============================================================
    print("\n【模型训练】")
    
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  训练 {name}...")
        
        model.fit(X_train_scaled, y_train)
        y_pred_log = model.predict(X_test_scaled)
        
        # 反变换
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    # ============================================================
    # 7. 选择最佳模型
    # ============================================================
    print("\n【模型对比】")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    
    print(f"  最佳模型: {best_model_name} (R²={results[best_model_name]['r2']:.4f})")
    
    # ============================================================
    # 8. 区域潜力预测
    # ============================================================
    print("\n【区域潜力预测】")
    
    # 构建所有省份-品类组合的潜力预测
    all_provinces = df_merged['所在省份'].unique()
    all_categories = df_merged['商品大类'].unique()
    
    # 创建空白市场预测数据
    potential_data = []
    
    for province in all_provinces:
        for category in all_categories:
            # 查找该省份-品类的历史数据
            historical = agg_data[
                (agg_data['所在省份'] == province) & 
                (agg_data['商品大类'] == category)
            ]
            
            if len(historical) > 0:
                # 已有历史数据，使用历史平均值
                avg_sales = historical['销售额'].mean()
                avg_quantity = historical['销量'].mean()
                status = '已覆盖'
            else:
                # 空白市场，使用模型预测
                # 构建特征
                province_data = region_stats[region_stats['所在省份'] == province]
                category_data = category_stats[category_stats['商品大类'] == category]
                
                if len(province_data) > 0 and len(category_data) > 0:
                    features = np.array([
                        le_province.transform([str(province)])[0] if province in le_province.classes_ else 0,
                        le_category.transform([str(category)])[0] if category in le_category.classes_ else 0,
                        6,  # 假设旺季月份
                        province_data['省份总销量'].values[0],
                        province_data['省份总销售额'].values[0],
                        province_data['省份客户数'].values[0],
                        province_data['省份订单数'].values[0],
                        province_data['省份覆盖品类'].values[0],
                        province_data['省份覆盖商品'].values[0],
                        province_data['省份活跃度'].values[0],
                        province_data['省份渗透率'].values[0],
                        category_data['品类总销量'].values[0],
                        category_data['品类总销售额'].values[0],
                        category_data['品类覆盖省份'].values[0],
                        category_data['品类平均单价'].values[0],
                        category_data['品类热度'].values[0],
                        category_data['品类渗透率'].values[0],
                        province_data['省份渗透率'].values[0] * category_data['品类渗透率'].values[0],
                        1,  # 旺季
                        0   # 非促销季
                    ])
                    
                    features_scaled = scaler.transform([features])
                    pred_log = best_model.predict(features_scaled)
                    pred_sales = np.expm1(pred_log[0])
                    
                    avg_sales = pred_sales
                    avg_quantity = pred_sales / (category_data['品类平均单价'].values[0] + 1)
                    status = '空白-预测'
                else:
                    avg_sales = 0
                    avg_quantity = 0
                    status = '数据不足'
            
            potential_data.append({
                '省份': province,
                '品类': category,
                '预测销售额': avg_sales,
                '预测销量': avg_quantity,
                '状态': status
            })
    
    potential_df = pd.DataFrame(potential_data)
    
    # 添加潜力等级
    potential_df['潜力等级'] = potential_df['预测销售额'].apply(
        lambda x: '🔥 高潜力' if x >= potential_df['预测销售额'].quantile(0.75) else
                  '✅ 中潜力' if x >= potential_df['预测销售额'].quantile(0.50) else
                  '⚠️ 低潜力' if x >= potential_df['预测销售额'].quantile(0.25) else '❌ 建议放弃'
    )
    
    # 空白市场建议
    blank_potential = potential_df[potential_df['状态'] == '空白-预测'].sort_values('预测销售额', ascending=False)
    
    print(f"\n  空白市场数: {len(blank_potential)}")
    print(f"  高潜力空白市场: {len(blank_potential[blank_potential['潜力等级'] == '🔥 高潜力'])}")
    
    # 显示TOP空白市场
    print("\n  TOP 10 空白市场潜力预测:")
    for i, row in blank_potential.head(10).iterrows():
        print(f"    {row['省份']} - {row['品类']}: 预测销售额 ¥{row['预测销售额']:.0f}, {row['潜力等级']}")
    
    # ============================================================
    # 9. 可视化
    # ============================================================
    print("\n【可视化】")
    
    charts_dir = os.path.join(base_dir, 'model_charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # 9a. 模型对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    r2_scores = [results[n]['r2'] for n in model_names]
    
    bars = ax.bar(model_names, r2_scores, color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'])
    ax.set_ylabel('R² Score')
    ax.set_title('模型对比 - R² Score')
    ax.set_ylim(0, 1)
    
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'region_potential_model_comparison.png'), dpi=150)
    plt.close()
    print(f"  保存: {charts_dir}/region_potential_model_comparison.png")
    
    # 9b. 空白市场潜力分布图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    potential_counts = blank_potential['潜力等级'].value_counts()
    
    ax.bar(potential_counts.index, potential_counts.values, 
           color=['#e74c3c', '#f39c12', '#2ecc71', '#95a5a6'])
    ax.set_title('空白市场潜力等级分布')
    ax.set_ylabel('市场数')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'blank_market_potential_distribution.png'), dpi=150)
    plt.close()
    print(f"  保存: {charts_dir}/blank_market_potential_distribution.png")
    
    # 9c. TOP空白市场热力图（省份-品类）
    if len(blank_potential) >= 5:
        pivot_data = blank_potential.pivot_table(
            values='预测销售额', 
            index='省份', 
            columns='品类', 
            aggfunc='sum'
        ).fillna(0)
        
        # 只显示TOP省份和品类
        top_provinces = blank_potential.groupby('省份')['预测销售额'].sum().nlargest(8).index
        top_categories = blank_potential.groupby('品类')['预测销售额'].sum().nlargest(6).index
        
        pivot_filtered = pivot_data.loc[
            pivot_data.index.intersection(top_provinces),
            pivot_data.columns.intersection(top_categories)
        ]
        
        if len(pivot_filtered) > 0 and len(pivot_filtered.columns) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(pivot_filtered.values, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(range(len(pivot_filtered.columns)))
            ax.set_yticks(range(len(pivot_filtered.index)))
            ax.set_xticklabels(pivot_filtered.columns, rotation=45, ha='right')
            ax.set_yticklabels(pivot_filtered.index)
            
            ax.set_title('空白市场潜力热力图（省份-品类）')
            plt.colorbar(im, ax=ax, label='预测销售额')
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'blank_market_heatmap.png'), dpi=150)
            plt.close()
            print(f"  保存: {charts_dir}/blank_market_heatmap.png")
    
    # ============================================================
    # 10. 保存结果
    # ============================================================
    print("\n【保存结果】")
    
    output_dir = os.path.join(base_dir, 'model_metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    joblib.dump(best_model, os.path.join(model_dir, 'region_sales_potential_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'region_potential_scaler.joblib'))
    joblib.dump(le_province, os.path.join(model_dir, 'province_encoder.joblib'))
    joblib.dump(le_category, os.path.join(model_dir, 'category_encoder.joblib'))
    print(f"  保存模型: {model_dir}/region_sales_potential_model.joblib")
    
    # 保存特征列
    joblib.dump(feature_cols, os.path.join(model_dir, 'region_potential_features.joblib'))
    
    # 保存区域和品类统计
    joblib.dump(region_stats, os.path.join(model_dir, 'region_stats.joblib'))
    joblib.dump(category_stats, os.path.join(model_dir, 'category_stats.joblib'))
    
    # 保存指标
    metrics = {name: {'rmse': r['rmse'], 'mae': r['mae'], 'r2': r['r2']} for name, r in results.items()}
    with open(os.path.join(output_dir, 'region_potential_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  保存指标: {output_dir}/region_potential_metrics.json")
    
    # 保存潜力预测结果
    potential_df.to_csv(os.path.join(output_dir, 'region_potential_prediction.csv'), 
                        index=False, encoding='utf-8-sig')
    print(f"  保存预测: {output_dir}/region_potential_prediction.csv")
    
    # 保存空白市场建议
    blank_potential.to_csv(os.path.join(output_dir, 'blank_market_suggestions.csv'), 
                           index=False, encoding='utf-8-sig')
    print(f"  保存空白市场: {output_dir}/blank_market_suggestions.csv")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"  R²: {results[best_model_name]['r2']:.4f}")
    print(f"  RMSE: {results[best_model_name]['rmse']:.2f}")
    
    print(f"\n空白市场发现:")
    print(f"  总空白市场: {len(blank_potential)}")
    print(f"  高潜力: {len(blank_potential[blank_potential['潜力等级'] == '🔥 高潜力'])}")
    print(f"  中潜力: {len(blank_potential[blank_potential['潜力等级'] == '✅ 中潜力'])}")
    
    return {
        'best_model': best_model_name,
        'r2': results[best_model_name]['r2'],
        'blank_count': len(blank_potential),
        'high_potential': len(blank_potential[blank_potential['潜力等级'] == '🔥 高潜力'])
    }


if __name__ == '__main__':
    main()