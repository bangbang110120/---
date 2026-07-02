# -*- coding: utf-8 -*-
"""
套餐组合推荐模型 - 商品聚类 + 协同购买分析
通过聚类算法发现相似商品群，再结合购买数据推荐套餐组合
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("套餐组合推荐模型 - 商品聚类分析")
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
    # 2. 构建商品特征矩阵（基于购买行为）
    # ============================================================
    print("\n【构建商品特征矩阵】")

    # 合并商品信息
    df_merged = df_sales.merge(df_product, on='商品编号', how='left')

    # 2a. 商品基本属性（在原始销售数据上聚合，避免合并后类型变化）
    df_sales['订购数量'] = pd.to_numeric(df_sales['订购数量'], errors='coerce').fillna(0)
    df_sales['金额'] = pd.to_numeric(df_sales['金额'], errors='coerce').fillna(0)

    product_stats = df_sales.groupby('商品编号').agg(
        总销量=('订购数量', 'sum'),
        订单数=('订单编码', 'nunique'),
        总金额=('金额', 'sum')
    ).reset_index()

    # 计算平均单价
    product_stats['平均单价'] = product_stats['总金额'] / product_stats['总销量'].replace(0, 1)

    product_info = df_product[['商品编号', '商品名称', '商品小类', '商品大类', '销售单价']].copy()
    product_stats = product_stats.merge(product_info, on='商品编号', how='left')

    # 2b. 构建购买共现矩阵
    order_products = df_merged.groupby('订单编码')['商品小类'].apply(list).reset_index()
    order_products = order_products[order_products['商品小类'].apply(len) >= 2]

    all_categories = sorted(df_product['商品小类'].dropna().unique())
    cat_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    n_cats = len(all_categories)

    co_occurrence = np.zeros((n_cats, n_cats))
    for cats in order_products['商品小类']:
        valid = [c for c in cats if isinstance(c, str)]
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                if valid[i] in cat_to_idx and valid[j] in cat_to_idx:
                    co_occurrence[cat_to_idx[valid[i]], cat_to_idx[valid[j]]] += 1
                    co_occurrence[cat_to_idx[valid[j]], cat_to_idx[valid[i]]] += 1

    print(f"  商品小类数: {n_cats}")
    print(f"  共现订单数: {len(order_products)}")

    # 2c. 每个商品的特征向量 = 它所在小类与其他小类的共现频率
    category_features = {}
    for cat in all_categories:
        idx = cat_to_idx[cat]
        vec = co_occurrence[idx, :] / (co_occurrence[idx, :].sum() + 1)
        category_features[cat] = vec

    X_cat = np.array([category_features[cat] for cat in all_categories])

    # ============================================================
    # 3. K-Means 聚类
    # ============================================================
    print("\n【K-Means 聚类】")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cat)

    # 确定最佳聚类数（肘部法则）
    inertias = []
    K_range = range(2, min(10, n_cats))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # 自动选K（简单启发式：找拐点）
    best_k = 4
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        best_idx = np.argmax(diffs2) + 2
        best_k = best_idx + 2 if best_idx + 2 >= 3 else 4

    print(f"  自动选择聚类数: K={best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # 分配到各小类
    cat_clusters = {cat: labels[i] for i, cat in enumerate(all_categories)}
    print(f"  各簇商品数: {dict(sorted(Counter(labels).items()))}")

    # ============================================================
    # 4. 聚类结果分析
    # ============================================================
    print("\n【聚类结果分析】")

    cluster_info = {}
    for cluster_id in range(best_k):
        cluster_cats = [cat for cat, lab in cat_clusters.items() if lab == cluster_id]
        
        # 簇内共现强度
        cluster_co_count = 0
        for c1 in cluster_cats:
            for c2 in cluster_cats:
                if c1 < c2 and c1 in cat_to_idx and c2 in cat_to_idx:
                    cluster_co_count += co_occurrence[cat_to_idx[c1], cat_to_idx[c2]]

        # 统计该簇商品的销量信息
        cluster_products = df_product[df_product['商品小类'].isin(cluster_cats)]
        cluster_sales = product_stats[product_stats['商品小类'].isin(cluster_cats)]

        cluster_info[cluster_id] = {
            '品类数': len(cluster_cats),
            '品类列表': cluster_cats,
            '商品数': len(cluster_products),
            '总销量': int(cluster_sales['总销量'].sum()),
            '簇内共现强度': int(cluster_co_count),
            '平均单价': round(cluster_sales['平均单价'].mean(), 1)
        }

        print(f"\n  簇 {cluster_id}: 包含 {len(cluster_cats)} 个小类")
        print(f"    品类: {cluster_cats}")
        print(f"    总销量: {cluster_info[cluster_id]['总销量']:,}")
        print(f"    簇内共现强度: {cluster_info[cluster_id]['簇内共现强度']:,}")

    # ============================================================
    # 5. 套餐组合推荐
    # ============================================================
    print("\n【套餐组合推荐】")

    # 获取商品单价
    product_prices = df_product.set_index('商品小类')['销售单价'].to_dict()

    bundle_suggestions = []
    bundle_id = 0

    for cluster_id in range(best_k):
        cluster_cats = [cat for cat, lab in cat_clusters.items() if lab == cluster_id]

        # 在每个簇内找最常一起购买的商品组合
        cluster_co_pairs = []
        for i in range(len(cluster_cats)):
            for j in range(i+1, len(cluster_cats)):
                if cluster_cats[i] in cat_to_idx and cluster_cats[j] in cat_to_idx:
                    count = co_occurrence[cat_to_idx[cluster_cats[i]], cat_to_idx[cluster_cats[j]]]
                    if count > 0:
                        cluster_co_pairs.append((cluster_cats[i], cluster_cats[j], count))

        # 按共现次数排序，取TOP组合
        cluster_co_pairs.sort(key=lambda x: -x[2])

        # 每个簇生成2-3个套餐
        seen_pairs = set()
        for pair_idx, (cat1, cat2, co_count) in enumerate(cluster_co_pairs):
            if bundle_id >= 20:
                break

            pair_key = tuple(sorted([cat1, cat2]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # 获取具体商品
            prods_cat1 = df_product[df_product['商品小类'] == cat1]
            prods_cat2 = df_product[df_product['商品小类'] == cat2]

            if len(prods_cat1) > 0 and len(prods_cat2) > 0:
                name1 = prods_cat1.iloc[0]['商品名称']
                name2 = prods_cat2.iloc[0]['商品名称']
                price1 = product_prices.get(cat1, 50)
                price2 = product_prices.get(cat2, 50)
                original = price1 + price2

                # 折扣策略：共现越强，折扣越小(说明刚需强)
                total_orders = max(order_products.shape[0], 1)
                pair_support = co_count / total_orders

                if pair_support >= 0.03:
                    discount = 0.85
                elif pair_support >= 0.01:
                    discount = 0.80
                else:
                    discount = 0.75

                bundle_id += 1
                bundle_name = f"{name1}+{name2}组合"

                suggestion = {
                    'bundle_id': bundle_id,
                    'bundle_name': bundle_name,
                    'items': [cat1, cat2],
                    'item_names': [name1, name2],
                    'cluster_id': int(cluster_id),
                    'co_occurrence': int(co_count),
                    'support': round(pair_support, 4),
                    'original_price': round(original, 1),
                    'bundle_price': round(original * discount, 1),
                    'discount_rate': f"{int((1-discount)*100)}%",
                    'recommendation': '强烈推荐' if pair_support >= 0.03 else '推荐' if pair_support >= 0.01 else '常规'
                }
                bundle_suggestions.append(suggestion)

                print(f"\n  套餐 #{bundle_id}: {bundle_name}")
                print(f"    组合: {cat1} + {cat2}")
                print(f"    共现次数: {co_count}")
                print(f"    原价: {original:.0f}元 → 套餐价: {original*discount:.0f}元 ({suggestion['discount_rate']})")
                print(f"    推荐: {suggestion['recommendation']}")

    # ============================================================
    # 6. 可视化
    # ============================================================
    print("\n【可视化】")

    charts_dir = os.path.join(base_dir, 'model_charts')
    os.makedirs(charts_dir, exist_ok=True)

    # PCA降维可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 聚类散点图
    ax = axes[0]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    for k in range(best_k):
        mask = labels == k
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[k % 4], label=f'簇{k}', s=100, alpha=0.7)
        # 标注品类名
        for i, cat in enumerate(all_categories):
            if labels[i] == k:
                ax.annotate(cat, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.8)

    ax.set_title('商品小类聚类结果 (PCA降维)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()

    # 肘部法则图
    ax = axes[1]
    ax.plot(list(K_range), inertias, 'b-o', markersize=8)
    ax.axvline(x=best_k, color='r', linestyle='--', label=f'最佳K={best_k}')
    ax.set_xlabel('聚类数 K')
    ax.set_ylabel('簇内平方和 (Inertia)')
    ax.set_title('肘部法则 - 最佳聚类数选择')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'product_clustering.png'), dpi=150)
    plt.close()
    print(f"  保存: {charts_dir}/product_clustering.png")

    # 共现热力图（TOP品类）
    top_cats = sorted(all_categories, key=lambda c: co_occurrence[cat_to_idx[c]].sum(), reverse=True)[:12]
    top_idx = [cat_to_idx[c] for c in top_cats]
    co_matrix_top = co_occurrence[np.ix_(top_idx, top_idx)]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(co_matrix_top, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(top_cats)))
    ax.set_yticks(range(len(top_cats)))
    ax.set_xticklabels(top_cats, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(top_cats, fontsize=9)
    ax.set_title('TOP12 品类购买共现热力图')
    plt.colorbar(im, ax=ax, label='共现次数')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'co_occurrence_heatmap.png'), dpi=150)
    plt.close()
    print(f"  保存: {charts_dir}/co_occurrence_heatmap.png")

    # ============================================================
    # 7. 保存结果
    # ============================================================
    print("\n【保存结果】")

    output_dir = os.path.join(base_dir, 'model_metrics')
    os.makedirs(output_dir, exist_ok=True)

    # 聚类信息
    for k, v in cluster_info.items():
        v['品类列表'] = [str(c) for c in v['品类列表']]

    with open(os.path.join(output_dir, 'product_clusters.json'), 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)
    print(f"  保存聚类结果: {output_dir}/product_clusters.json")

    # 套餐建议
    with open(os.path.join(output_dir, 'bundle_suggestions.json'), 'w', encoding='utf-8') as f:
        json.dump(bundle_suggestions, f, ensure_ascii=False, indent=2)
    print(f"  保存套餐建议: {output_dir}/bundle_suggestions.json")

    # 模型文件
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    import joblib
    joblib.dump(km, os.path.join(model_dir, 'product_clustering_model.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'product_clustering_scaler.joblib'))
    joblib.dump(cat_to_idx, os.path.join(model_dir, 'category_index.joblib'))
    joblib.dump(all_categories, os.path.join(model_dir, 'all_categories.joblib'))
    print(f"  保存模型: {model_dir}/product_clustering_model.joblib")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    print(f"\n聚类数: {best_k}")
    print(f"套餐建议数: {len(bundle_suggestions)}")

    print("\n输出文件:")
    print(f"  - {output_dir}/product_clusters.json")
    print(f"  - {output_dir}/bundle_suggestions.json")
    print(f"  - {charts_dir}/product_clustering.png")
    print(f"  - {charts_dir}/co_occurrence_heatmap.png")

    return {
        'clusters': best_k,
        'bundles': len(bundle_suggestions),
        'top_bundles': bundle_suggestions[:5]
    }


if __name__ == '__main__':
    main()
