import matplotlib.pyplot as plt
import numpy as np
def draw_hist(x,bins=30,density=False):
    # 直方图绘制
    # hist给定一个数组之类的对象，自动计算相关频数或频率
    # bins表示分多少组，range表示显示的范围，align对齐方式left’, ‘mid’, ‘right，density表示True即求频率

    plt.subplots()
    plt.hist(x,
             bins=bins,
             # range=(50, 140),
             density=density,
             align=u'left'
             )
    plt.show()


def analyze_unordered_numerical_array(data,bins=10,density=False):
    """
    分析数值数组:无序的数据，计算频率
    Args:
        data: 输入的是数据的列表，所有数据间无序，计算频数以及直方图

    Returns:

    """

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
        '计数':{len(data)},
        '均值':      {data.mean()},
    '标准差':   {data.std()},
    '方差':     {data.var()},
    '最大值':    {np.max(data)},
    '最小值':    {np.min(data)},
    '下四分位数': {q1},
    '中位数':  {q2},
    '上四分位数': {q3},
    '下异常值界限':{lower_limit}, '下异常值数': {len(np.where(data < lower_limit)[0])},
    '上异常值界限':{upper_limit}, '上异常值数': {len(np.where(data > upper_limit)[0])},
        '可能错误的数据':wrong_data
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




if __name__ == '__main__':

    x = 100 + 15 * np.random.randn(437)
    # print(np.percentile(x, 25))
    # draw_hist(x)
    x =[1,2,4,5,6,7,8,9,10]
    analyze_unordered_numerical_array(x,bins=30)