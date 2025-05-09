import pandas as pd

def replace_csv_paths(csv_path, old_root, new_root, output_csv_path):
    df = pd.read_csv(csv_path)

    # 替换 prior 和 img 字段中的路径前缀
    df['prior'] = df['prior'].str.replace(old_root, new_root, regex=False)
    df['img'] = df['img'].str.replace(old_root, new_root, regex=False)

    # 保存新 CSV
    df.to_csv(output_csv_path, index=False)
    print(f"✅ 已替换路径前缀，保存到：{output_csv_path}")

# 示例调用
csv_path = 'tutorial2_val_prior.csv'
output_csv_path = 'tutorial2_val_prior.csv'
old_root = 'E:\\Projects\\yang_proj\data\\seg2med\\'
new_root = ''


replace_csv_paths(csv_path, old_root, new_root, output_csv_path)
