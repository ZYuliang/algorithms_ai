def show_plt_completely():
    """
        plt显示问题
    Returns:

    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号，解决图像中的“-”负号的乱码问题
