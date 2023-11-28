def show_dataframe_completely():
    """
    完全显示pandas的dataframe的所有值
    Returns:

    """
    import pandas as pd
    pd.set_option('max_colwidth', 500)  # 设置value的显示长度为200，默认为50
    pd.set_option('display.max_columns', None)  # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_rows', None)  # 显示所有行，把列显示设置成最大


def get_data_structure(data):
    if isinstance(data, str):
        return 'str\n'
    elif isinstance(data, list):
        if len(data) >= 1:
            return f'list({len(data)}:\n' + get_data_structure(data[0]) + ')\n'
        else:
            return f'list{0}:\n'
    elif isinstance(data, dict):
        if not data:
            return 'dict\n'

        r = 'dict(\n'
        for i, j in data.items():
            r += f'{i}:'
            r += get_data_structure(j)
        r += ')\n'
        return r
    elif isinstance(data, bool):
        return 'bool\n'
    else:
        if not data:
            return 'None\n'
        return 'other\n'


def show_data_structure(data):
    data_structure = get_data_structure(data)
    data_structure = data_structure.split('\n')
    index = 0
    step = 4
    for i in data_structure:
        if '(' in i:
            print(index * ' ' + i)
            index += step
        elif ')' in i:
            index = index - step
            print(index * ' ' + i)
        else:
            print(index * ' ' + i)


import numpy as np


def smooth(x, y, box_pts, remove_bound=False):
    # 平滑数据,使用的是一维卷积
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    if remove_bound:
        if box_pts == 2:
            x = x[box_pts // 2::]
            y_smooth = y_smooth[box_pts // 2::]
        else:
            x = x[box_pts // 2: -(box_pts // 2)]
            y_smooth = y_smooth[box_pts // 2: -(box_pts // 2)]
    return x, [round(i, 2) for i in y_smooth]
