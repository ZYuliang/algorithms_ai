"""
使用mpire进行多进程多线程处理
"""
from mpire import WorkerPool
from mpire.utils import make_single_arguments


def apply_mpire(func, data_list, job_num=4, **shared_para):
    # 使用mpire做for循环函数的加速处理，需要输入函数和数据，以及相应的共享参数
    def refined_func(shared_para, x):
        return func(x, **shared_para)

    if shared_para:
        shared_objects = shared_para
    else:
        shared_objects = None
    if isinstance(data_list[0], dict):
        with WorkerPool(n_jobs=job_num, shared_objects=shared_objects) as pool:
            results = pool.map(refined_func, make_single_arguments(data_list, generator=False), progress_bar=True)
    else:
        with WorkerPool(n_jobs=job_num, shared_objects=shared_objects) as pool:
            results = pool.map(refined_func, data_list, progress_bar=True)
    return results
