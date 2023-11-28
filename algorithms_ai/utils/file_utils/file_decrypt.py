import pandas as pd


def file_decrypt(file, to_file_name, sheet_name=None):
    # 文件解密,读取并存在哪
    if 'json' in file:
        dt = pd.read_json(file)
        dt.to_json(to_file_name)
    else:
        if not sheet_name:
            dt = pd.read_excel(file)
        else:
            dt = pd.read_excel(file, sheet_name=sheet_name)
        print('read')
        dt.to_excel(to_file_name)
        print('success')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='file_decrypt')
    parser.add_argument('--file_name', help='file_name')
    parser.add_argument('--to_file_name', help='to_file_name')
    parser.add_argument('--sheet_name', help='sheet_name', default=None)
    args = parser.parse_args()
    file_decrypt(args.file_name, args.to_file_name, args.sheet_name)

    # f =
    # file_decrypt()