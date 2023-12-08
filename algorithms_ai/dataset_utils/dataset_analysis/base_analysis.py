"""
基本的分析，

matplotlib用3.6.2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('max_colwidth', 500)  # 设置value的显示长度为200，默认为50
pd.set_option('display.max_columns', None)  # 显示所有列，把行显示设置成最大
pd.set_option('display.max_rows', None)  # 显示所有行，把列显示设置成最大

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from algorithms_ai.utils.string_utils.string_utils import get_text_language

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('max_colwidth', 500)  # 设置value的显示长度为200，默认为50
pd.set_option('display.max_columns', None)  # 显示所有列，把行显示设置成最大
pd.set_option('display.max_rows', None)  # 显示所有行，把列显示设置成最大


def analyze_numerical_array(data):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    q1 = np.percentile(data, 25)  # 第一四分位数，从小到大25%,下四分位数
    q2 = np.percentile(data, 50)  # 第二四分位数，从小到大50%，中位数
    q3 = np.percentile(data, 75)  # 第三四分位数，从小到大75%，上四分位数
    iqr = q3 - q1  # 四分位数差（IQR，interquartile range），上四分位数-下四分位数
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    print(f"""
      计数：      {len(data)}
      均值：      {data.mean()}
      标准差：     {data.std()}
      方差：      {data.var()}
      最大值：    {np.max(data)}
      最小值：    {np.min(data)}
      下四分位数： {q1}
      中位数：     {q2}
      上四分位数:  {q3}
      下异常值界限：{lower_limit}   ,异常值数:{len(np.where(data < lower_limit)[0])}
      上异常值界限：{upper_limit}   ,异常值数:{len(np.where(data > upper_limit)[0])}
          """
          )

    plt.subplot(211)
    plt.hist(data)
    plt.subplot(212)
    plt.boxplot(data, vert=False)
    plt.show()


def analyze_category_array(data: pd.Series):
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    data_value_counts = data.value_counts()
    data_pie = data_value_counts / len(data)
    print(f"""
      data: 
      {data_value_counts}
      data_percent:
      {data_pie.sort_values}
      """
          )
    plt.subplot()
    data_value_counts.plot.bar()
    plt.show()
    plt.subplot()
    data_pie.plot.pie(autopct='%.1f%%', title='pie', )
    plt.show()


def analyze_unordered_numerical_array(data, bins=10, density=False):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    q1 = np.percentile(data, 25)  # 第一四分位数，从小到大25%,下四分位数
    q2 = np.percentile(data, 50)  # 第二四分位数，从小到大50%，中位数
    q3 = np.percentile(data, 75)  # 第三四分位数，从小到大75%，上四分位数
    iqr = q3 - q1  # 四分位数差（IQR，interquartile range），上四分位数-下四分位数
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    wrong_data = []
    wrong_data.extend([data[i] for i in np.where(data < lower_limit)])
    wrong_data.extend([data[i] for i in np.where(data > upper_limit)])
    results = {
        '计数': {len(data)},
        '均值': {data.mean()},
        '标准差': {data.std()},
        '方差': {data.var()},
        '最大值': {np.max(data)},
        '最小值': {np.min(data)},
        '下四分位数': {q1},
        '中位数': {q2},
        '上四分位数': {q3},
        '下异常值界限': {lower_limit}, '下异常值数': {len(np.where(data < lower_limit)[0])},
        '上异常值界限': {upper_limit}, '上异常值数': {len(np.where(data > upper_limit)[0])},
        # '可能错误的数据': wrong_data
    }

    print(results)

    plt.subplot(211)
    plt.hist(data,
             bins=bins,
             # range=(50, 140),
             density=density,
             align=u'left'
             )
    # plt.axvline(data.mean(), color='r')

    plt.subplot(212)
    plt.boxplot(data, vert=False)
    plt.show()
    return results


def draw_hist(x, bins=30, density=False):
    plt.subplots()
    plt.hist(x,
             bins=bins,
             # range=(50, 140),
             density=density,
             align=u'left'
             )
    plt.show()


def analyze_texts_list(texts,
                       tokenizer=None,
                       analyze_char_length=False,
                       analyze_text_language=False,
                       analyze_token_length=False,
                       ):
    if analyze_char_length:
        print('输入的字符长度：')
        texts_length = [len(texts) for i in texts]
        analyze_numerical_array(texts_length)

    if analyze_text_language:
        print('输入的文本语言：')
        texts_language = [get_text_language(i) for i in texts]
        analyze_category_array(texts_language)

    if tokenizer:
        tokens = [tokenizer(i)['texts'] for i in texts]

        if analyze_token_length:
            print('输入的token长度：')
            token_length = [len(i) for i in tokens]
            analyze_numerical_array(token_length)


if __name__ == '__main__':
    texts = [
        'A Phase 2 Multi-center, Open-label, Single Arm Study of Nab-sirolimus in Patients With Well-differentiated Neuroendocrine Tumors (NETs) of the Gastrointestinal Tract, Lung, or Pancreas Who Have Not Received Prior Treatment With mTOR Inhibitors',
        'A Phase 2 Multi-center, Open-label, Single Arm Study of Nab-sirolimus in Patients With Well-differentiated Neuroendocrine Tumors (NETs) of the Gastrointestinal Tract, Lung, or Pancreas Who Have Not Received Prior Treatment With mTOR Inhibitors',
        'A Phase 2 Multi-center, Open-label, Single Arm Study of Nab-sirolimus in Patients With Well-differentiated Neuroendocrine Tumors (NETs) of the Gastrointestinal Tract, Lung, or Pancreas Who Have Not Received Prior Treatment With mTOR Inhibitors']
    z = list(range(30))
    from algorithms_ai.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer

    analyze_texts_list(texts, RegexTokenizer(),
                       analyze_token_length=True
                       )
