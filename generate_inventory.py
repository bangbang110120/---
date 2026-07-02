"""
纯Python版本：基于B2B日化数据和淘宝美妆数据生成库存预测表。
避免pandas/numpy兼容性问题。
"""
import csv
import random
from collections import defaultdict
from datetime import date, timedelta

random.seed(42)

TODAY = date(2025, 12, 1)  # 假设双十一后盘点
DATA_START = date(2025, 4, 1)
DATA_END = date(2025, 11, 30)
TOTAL_DAYS = (DATA_END - DATA_START).days  # 243天

# ===================== 读取B2B数据 =====================
print("读取B2B数据...")
rihua_rows = []
with open('cleaned_data/rihua_merged.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rihua_rows.append(row)

# 按商品汇总
prod_data = defaultdict(lambda: {
    'name': '', 'sub_cat': '', 'cat': '', 'total_qty': 0, 'order_count': 0,
    'unit_price': 0, 'last_date': None, 'prices': []
})
for row in rihua_rows:
    pid = row.get('商品编号', '').strip()
    if not pid or not pid.startswith('X'):
        continue
    if not pid[1:].isdigit():
        continue
    qty = float(row.get('订购数量', 0))
    price = float(row.get('销售单价', 0))
    order_date = row.get('订单日期', '')
    sub_cat = row.get('商品小类', '')
    cat = row.get('商品大类', '')
    name = row.get('商品名称', '')

    if not sub_cat or not cat:
        continue

    d = prod_data[pid]
    d['name'] = name
    d['sub_cat'] = sub_cat
    d['cat'] = cat
    d['total_qty'] += qty
    d['order_count'] += 1
    if price > 0:
        d['prices'].append(price)
    if order_date:
        try:
            dt = date.fromisoformat(order_date)
            if d['last_date'] is None or dt > d['last_date']:
                d['last_date'] = dt
        except ValueError:
            pass

# 计算派生指标
products = []
for pid, d in prod_data.items():
    active_days = max(60, (d['last_date'] - DATA_START).days + 1 if d['last_date'] else TOTAL_DAYS)
    avg_daily = d['total_qty'] / active_days
    avg_price = sum(d['prices']) / len(d['prices']) if d['prices'] else 0
    products.append({
        'pid': pid,
        'name': d['name'],
        'sub_cat': d['sub_cat'],
        'cat': d['cat'],
        'total_qty': int(d['total_qty']),
        'order_count': d['order_count'],
        'avg_price': round(avg_price, 2),
        'avg_daily': round(avg_daily, 1),
        'monthly_qty': int(avg_daily * 30),
        'active_days': active_days,
    })

print(f"B2B商品数: {len(products)}")

# ===================== 模拟库存 =====================
n = len(products)
avg_dailies = [p['avg_daily'] for p in products]

# 库存覆盖天数 - 偏态分布，部分商品设为低位
coverage_days = [random.uniform(15, 120) for _ in range(n)]
# 前20%设为偏低
n_low = max(3, n // 5)
low_indices = random.sample(range(n), n_low)
for i in low_indices:
    coverage_days[i] = random.uniform(3, 14)
# 10%设为告急
remaining = [i for i in range(n) if i not in low_indices]
n_critical = max(2, n // 10)
critical_indices = random.sample(remaining, n_critical)
for i in critical_indices:
    coverage_days[i] = random.uniform(0.5, 3)

for i, p in enumerate(products):
    p['current_stock'] = max(5, int(p['avg_daily'] * coverage_days[i]))

# 安全库存
for p in products:
    if p['current_stock'] / max(p['avg_daily'], 1) < 30:
        safety_cover = 7
    else:
        safety_cover = 21
    p['safety_stock'] = max(3, int(p['avg_daily'] * safety_cover))

# 预计耗尽天数 & 库存状态
def classify_stock(days, qty, safety):
    if qty <= 0:
        return '已断货'
    if qty <= safety:
        return '告急⚠️'
    if days <= 7:
        return '偏低'
    if days <= 30:
        return '正常'
    return '充足'

for p in products:
    p['depletion_days'] = max(0, int((p['current_stock'] - p['safety_stock']) / max(p['avg_daily'], 0.1)))
    p['stock_status'] = classify_stock(p['depletion_days'], p['current_stock'], p['safety_stock'])

# 最近补货日期
restock_base = TODAY - timedelta(days=30)
for p in products:
    p['restock_date'] = (restock_base + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')

# 建议补货量
for p in products:
    if p['current_stock'] <= p['safety_stock']:
        target = p['avg_daily'] * 60 + p['safety_stock']
        p['restock_qty'] = max(0, int(target - p['current_stock']))
    elif p['depletion_days'] <= 14:
        p['restock_qty'] = int(p['avg_daily'] * 30)
    else:
        p['restock_qty'] = 0

# 仓库 & 库位
warehouses = ['华东仓-杭州', '华南仓-广州', '华北仓-天津', '西南仓-成都', '华中仓-武汉']
wh_weights = [35, 25, 20, 10, 10]
for p in products:
    p['warehouse'] = random.choices(warehouses, weights=wh_weights, k=1)[0]
    p['location'] = f"A-{random.randint(1,19):02d}-{random.randint(1,49):02d}"

suppliers = ['广东美妆供应链', '上海日化集团', '浙江日化代工', '江苏生物科技', '福建化妆品厂', '山东化工集团']
for p in products:
    p['supplier'] = random.choice(suppliers)

# ===================== 读取淘宝美妆数据，取TOP 30 =====================
print("读取淘宝美妆数据...")
beauty_rows = []
with open('cleaned_data/beauty_cleaned.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        beauty_rows.append(row)

# 按sale_count排序取TOP 30
beauty_rows_sorted = sorted(beauty_rows, key=lambda r: int(float(r.get('sale_count', 0))), reverse=True)
beauty_top = beauty_rows_sorted[:30]

beauty_products = []
for row in beauty_top:
    pid = 'TB-' + row.get('id', '000')
    name = row.get('title', '')[:50]
    cat = row.get('商品品类', '其他')
    sale_count = int(float(row.get('sale_count', 0)))
    price = float(row.get('price', 0))
    avg_daily = sale_count / random.uniform(30, 90)
    cov = random.uniform(10, 90)
    if random.random() < 0.17:
        cov = random.uniform(1, 10)
    current_stock = max(3, int(avg_daily * cov))
    safety_stock = max(2, int(avg_daily * 14))
    depl_days = max(0, int((current_stock - safety_stock) / max(avg_daily, 0.1)))
    status = classify_stock(depl_days, current_stock, safety_stock)
    restock_qty = 0
    if current_stock <= safety_stock:
        restock_qty = max(0, int(avg_daily * 45 + safety_stock - current_stock))
    elif depl_days <= 14:
        restock_qty = int(avg_daily * 20)

    beauty_products.append({
        'pid': pid, 'name': name, 'sub_cat': cat, 'cat': cat,
        'total_qty': sale_count, 'order_count': 0, 'avg_price': round(price, 2),
        'avg_daily': round(avg_daily, 1), 'monthly_qty': int(avg_daily * 30),
        'active_days': random.randint(30, 90),
        'current_stock': current_stock,
        'safety_stock': safety_stock,
        'depletion_days': depl_days,
        'stock_status': status,
        'restock_date': (restock_base + timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
        'restock_qty': restock_qty,
        'warehouse': random.choices(['电商仓-义乌', '电商仓-广州'], weights=[60, 40], k=1)[0],
        'location': f"TB-{random.randint(1,19):02d}-{random.randint(1,49):02d}",
        'supplier': random.choice(suppliers),
    })

print(f"淘宝商品数: {len(beauty_products)}")

# ===================== 合并输出 =====================
all_products = []
for p in products:
    all_products.append((p, 'B2B渠道'))
for p in beauty_products:
    all_products.append((p, '天猫旗舰店'))

# 自定义排序：告急>偏低>正常>充足
status_order = {'告急⚠️': 0, '偏低': 1, '正常': 2, '充足': 3, '已断货': 0}
all_products.sort(key=lambda x: (status_order.get(x[0]['stock_status'], 5), x[0]['depletion_days']))

# 写CSV
output_path = 'inventory_stock.csv'
with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        '商品编号', '商品名称', '商品小类', '商品大类',
        '历史总销量', '订单数',
        '当前库存量', '最近补货日期'
    ])

    for p, source in all_products:
        writer.writerow([
            p['pid'], p['name'], p['sub_cat'], p['cat'],
            p['total_qty'], p.get('order_count', 0),
            p['current_stock'], p['restock_date']
        ])

# ===================== 统计 =====================
print(f'\n库存表已生成: {output_path}')
print(f'总计 {len(all_products)} 条记录')
print(f'B2B渠道: {len(products)} 条, 天猫旗舰店: {len(beauty_products)} 条')

# 状态分布
status_count = defaultdict(int)
for p, _ in all_products:
    status_count[p['stock_status']] += 1
print('\n库存状态分布:')
for s in ['告急⚠️', '已断货', '偏低', '正常', '充足']:
    if status_count[s]:
        print(f'  {s}: {status_count[s]} 件')

# 品类汇总
cat_summary = defaultdict(lambda: {'count': 0, 'total_stock': 0, 'critical': 0})
for p, _ in all_products:
    c = cat_summary[p['cat']]
    c['count'] += 1
    c['total_stock'] += p['current_stock']
    if p['stock_status'] in ('告急⚠️', '已断货'):
        c['critical'] += 1
print('\n各品类库存汇总:')
for cat, c in sorted(cat_summary.items()):
    print(f'  {cat}: {c["count"]}件商品, 总库存{c["total_stock"]}, 告急{c["critical"]}件')

# 前10紧急
print('\n⚠️ 最紧急的10件商品:')
for p, source in all_products[:10]:
    print(f'  [{p["stock_status"]}] {p["pid"]} | {p["name"]} | {p["cat"]} | 库存:{p["current_stock"]} | 安全:{p["safety_stock"]} | 耗尽:{p["depletion_days"]}天 | {source}')

print('\n✅ 完成！')
