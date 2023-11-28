"""
函数时间限制装饰器 set_timeout
时间搜索（向前，向后，日月年，间隔，开始时间，结束时间）：

"""
import signal
from loguru import logger
import time

def set_timeout(num, callback):
    """
    函数限制时间的装饰器，需要有传入的时间（秒）和超时后的处理函数
    :param num: 传入的时间（秒）
    :param callback: 超时后的处理函数
    :return:
    def after_timeout():  # 超时后的处理函数
    print("Time out!")

    @set_timeout(2, after_timeout)  # 限时 2 秒超时
    def connect():  # 要执行的函数
        time.sleep(1)  # 函数执行时间，写大于2的值，可测试超时
        print('Finished without timeout.')
    """
    def wrap(func):
        def handle(signum, frame):  # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)  # 设置信号和回调函数
                signal.alarm(num)  # 设置 num 秒的闹钟
                print('start alarm signal.')
                r = func(*args, **kwargs)
                print('close alarm signal.')
                signal.alarm(0)  # 关闭闹钟
                return r
            except RuntimeError as e:
                callback()

        return to_do

    return wrap

def call_back_month(last_date, month=3, forward=True):
    last_date_split = last_date.split('-')
    end_year, end_month = last_date_split[0], last_date_split[1]

    if forward:
        recall_month = int(end_month) + month
        if recall_month >= 13:
            end_year = str(int(end_year) + recall_month // 12)
            recall_month = recall_month % 12

    else:
        recall_month = int(end_month) - month
        if recall_month <= 0:
            end_year = str(int(end_year) - (-recall_month) // 12 - 1)
            recall_month = 12 - (-recall_month) % 12
    recall_month = str(recall_month)
    if len(recall_month) == 1:
        recall_month = '0' + recall_month

    return end_year + '-' + recall_month


def get_time_keys(start_time, end_time, time_interval):
    time_keys = []
    e_time = end_time
    s_time = call_back_month(e_time, month=time_interval - 1, forward=False)

    while s_time >= start_time:
        time_keys.append(s_time + '~' + e_time)
        e_time = call_back_month(s_time, month=1, forward=False)
        s_time = call_back_month(e_time, month=time_interval - 1, forward=False)

    # if s_time < start_time and e_time >= start_time:
    #     s_time = start_time
    #     time_keys.append(s_time + '~' + e_time)

    return list(reversed(time_keys))

def get_current_time(mode = 'timestamp'):
    if mode == 'struct_time':  # 格式化的时间,struct_time格式
        return time.localtime(time.time())
    elif mode == 'str_time': # 格式化的
        return time.asctime(time.localtime(time.time()))
    else:  # 10位时间戳
        return time.time()

# time_str = "2022-03-18 10:54:00"
# struct_time = time.strptime(time_str, '%Y-%m-%d %H:%M:%S')
# time.mktime(time.strptime('2023-05-19', '%Y-%m-%d'))

def convert_timestamp2timestring(time_stamp,timestring_format='%Y-%m-%d %H:%M:%S'):
    # 时间戳 => 结构化时间 => 时间字符串
    return time.strftime(timestring_format, time.localtime(time_stamp))

def convert_timestring2timestamp(timestring, timestring_format='%Y-%m-%d %H:%M:%S'):
    # 时间字符串 => 结构化时间 => 时间戳
    return time.mktime(time.strptime(timestring, timestring_format))

    # time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_stamp)




def time_spend(func):
    # 查询函数运行时间的装饰器
    def func_in(*args, **kwargs):
        logger.info(f'start function:{func.__name__}')
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f'end function:{func.__name__}')
        logger.info(f'function：{func.__name__} ,spend time：{round(end_time - start_time, 4)} s')
        return res  # test函数的返回值

    return func_in




def get_run_time(func, *args, **kwargs):
    # 直接传入参数获取函数的运行时间
    start_time = time.time()
    res = func(*args, **kwargs)
    end_time = time.time()
    logger.info(f'function：{func.__name__} ,spend time：{round(end_time - start_time, 4)} s')
    return res



if __name__ == '__main__':
    import time
    # @set_timeout(2, after_timeout)  # 限时 2 秒超时
    # def connect():  # 要执行的函数
    #     time.sleep(1)  # 函数执行时间，写大于2的值，可测试超时
    #     print('Finished without timeout.')
    print(time.time())
    print(1)

