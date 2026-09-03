import pandas as pd
import json
import pathlib

bench = pathlib.Path('data/processed/benchmark')

# Đọc stats tổng
stats = pd.read_csv(bench / 'benchmark_stats.csv')
print(stats[['window_id', 'quarter_label', 'n_dga', 'n_benign', 'n_total']].to_string(index=False))

# Tổng toàn benchmark
print(f'\nTổng DGA   : {stats.n_dga.sum():>10,}')
print(f'Tổng Benign: {stats.n_benign.sum():>10,}')
print(f'Tổng domain: {stats.n_total.sum():>10,}')

# Drift labels
with open(bench / 'drift_labels.json') as f:
    labels = json.load(f)
print(f'\nSố ranh giới drift: {len(labels["boundaries"])} (mong đợi: 23)')

# Xem thêm một vài dòng đầu D01
print('\n--- 5 dòng đầu D01.csv ---')
d01 = pd.read_csv(bench / 'D01.csv')
print(d01.head())
print(f'\nCác gia đình DGA trong D01: {d01[d01.label==1].family.nunique()} gia đình')