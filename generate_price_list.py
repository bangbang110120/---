"""
从B2B和淘宝数据中提取商品价格表。
字段：商品编号, 商品名称, 商品小类, 商品大类, 当前售价, 当前库存量, 历史总销量, 是否套装
"""
import csv
import re
from collections import defaultdict

def is_giftbox(title):
    """判断是否套装/礼盒"""
    keywords = ['套装', '礼盒', '组合', '套组', 'set', '搭配', '礼包']
    title_lower = title.lower()
    for kw in keywords:
        if kw in title_lower:
            return 1
    return 0

# ===================== B2B商品 =====================
b2b_products = {}
with open('cleaned_data/rihua_merged.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row.get('商品编号', '').strip()
        if not pid or not pid.startswith('X') or not pid[1:].isdigit():
            continue
        sub_cat = row.get('商品小类', '').strip()
        cat = row.get('商品大类', '').strip()
        if not sub_cat or not cat:
            continue
        name = row.get('商品名称', '').strip()
        price = float(row.get('销售单价', 0))
        qty = float(row.get('订购数量', 0))
        if pid not in b2b_products:
            b2b_products[pid] = {
                'name': name, 'sub_cat': sub_cat, 'cat': cat,
                'total_qty': 0, 'prices': [], 'order_count': 0
            }
        b2b_products[pid]['total_qty'] += qty
        b2b_products[pid]['order_count'] += 1
        if price > 0:
            b2b_products[pid]['prices'].append(price)

# 取平均售价作为当前售价
for pid, d in b2b_products.items():
    if d['prices']:
        d['avg_price'] = round(sum(d['prices']) / len(d['prices']), 2)
    else:
        d['avg_price'] = 0
    d['total_qty'] = int(d['total_qty'])

# 从inventory_stock.csv读取库存量
inventory_stock = {}
with open('inventory_stock.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row['商品编号'].strip()
        inventory_stock[pid] = int(row['当前库存量'])

# ===================== 淘宝商品 =====================
beauty_products = []
with open('cleaned_data/beauty_cleaned.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = 'TB-' + row.get('id', '')
        name = row.get('title', '')[:50]
        cat = row.get('商品品类', '其他')
        price = float(row.get('price', 0))
        sale_count = int(float(row.get('sale_count', 0)))
        gift = is_giftbox(name)
        beauty_products.append({
            'pid': pid, 'name': name, 'sub_cat': cat, 'cat': cat,
            'price': round(price, 2), 'total_qty': sale_count,
            'order_count': 0, 'is_giftbox': gift
        })

# 去重（同一id取最新快照的价格，取最大sale_count）
by_id = {}
for p in beauty_products:
    pid = p['pid']
    if pid not in by_id or p['total_qty'] > by_id[pid]['total_qty']:
        by_id[pid] = p
beauty_products = sorted(by_id.values(), key=lambda x: x['total_qty'], reverse=True)

# 取TOP 30
beauty_top = beauty_products[:30]

# ===================== 合并输出 =====================
rows = []
for pid, d in b2b_products.items():
    stock = inventory_stock.get(pid, d['total_qty'] // 30)
    rows.append([
        pid, d['name'], d['sub_cat'], d['cat'],
        d['avg_price'], stock, d['total_qty'], 0  # B2B商品不是套装
    ])

for p in beauty_top:
    stock = inventory_stock.get(p['pid'], max(3, p['total_qty'] // 60))
    rows.append([
        p['pid'], p['name'], p['sub_cat'], p['cat'],
        p['price'], stock, p['total_qty'], p['is_giftbox']
    ])

# 按商品编号排序
def sort_key(r):
    pid = r[0]
    if pid.startswith('X'):
        # pad to 3 digits for correct sorting
        num = pid[1:]
        return (0, int(num) if num.isdigit() else 0)
    else:
        return (1, pid)

rows.sort(key=sort_key)

with open('product_price_list.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['商品编号', '商品名称', '商品小类', '商品大类', '当前售价', '当前库存量', '历史总销量', '是否套装'])
    for row in rows:
        writer.writerow(row)

# 统计
print(f'商品价格表已生成: product_price_list.csv')
print(f'总计: {len(rows)} 条')
print(f'  B2B渠道: {len(b2b_products)} 条')
print(f'  天猫旗舰店: {len(beauty_top)} 条')
print(f'  套装商品: {sum(r[7] for r in rows)} 个')
print(f'\n品类分布:')
cats = defaultdict(int)
for r in rows:
    cats[r[3]] += 1
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f'  {c}: {n} 个')
print(f'\n价格区间:')
prices = [r[4] for r in rows if r[4] > 0]
print(f'  最低: {min(prices):.0f} 元')
print(f'  最高: {max(prices):.0f} 元')
print(f'  均价: {sum(prices)/len(prices):.0f} 元')
