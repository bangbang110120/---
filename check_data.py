import pandas as pd

# 查看清洗后的日化销售数据
df_sales = pd.read_csv('d:/619/项目一/cleaned_data/rihua_merged.csv')
print('=== 清洗后日化销售数据 ===')
print(f'总记录数: {len(df_sales)}')
print(f'独特订单编码数: {df_sales["订单编码"].nunique()}')
print(f'独特客户编码数: {df_sales["客户编码"].nunique()}')

# 查看 RFM 数据
rfm = pd.read_csv('d:/619/项目一/cleaned_data/customer_rfm.csv')
print(f'\n=== RFM数据 ===')
print(f'客户数: {len(rfm)}')
print(f'\n客户层级分布:')
print(rfm['客户层级'].value_counts())

# 查看流失客户
print(f'\n流失风险客户数: {(rfm["客户层级"] == "流失风险").sum()}')