import numpy as np
import pandas as pd
from pathlib import Path

u_all = np.array([0.2], dtype=float)

src_x_list = np.round(np.arange(0.5, 7.5 + 1e-9, 0.125), 3)
src_y_list = np.round(np.arange(0.5, 2.5 + 1e-9, 0.125), 3)

rows = []
case_id = 900001
for inlet_u in u_all:
    for sx in src_x_list:
        for sy in src_y_list:
            rows.append({
                'case_id': f'{case_id:06d}',
                'inlet_u': float(inlet_u),
                'src_x': float(sx),
                'src_y': float(sy),
            })
            case_id += 1

df = pd.DataFrame(rows, columns=['case_id', 'inlet_u', 'src_x', 'src_y'])
out_path = Path(__file__).resolve().parent / 'cases.csv'
df.to_csv(out_path, index=False)

print(f'[OK] wrote {len(df)} cases to {out_path}')
print(f'x count = {len(src_x_list)}, y count = {len(src_y_list)}')
print(f'x range = [{src_x_list.min():.3f}, {src_x_list.max():.3f}] step={src_x_list[1]-src_x_list[0]:.3f}')
print(f'y range = [{src_y_list.min():.3f}, {src_y_list.max():.3f}] step={src_y_list[1]-src_y_list[0]:.3f}')
