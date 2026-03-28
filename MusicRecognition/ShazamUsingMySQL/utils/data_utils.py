from utils.print_utils import print_error, print_message, print_warning
import time


class ProcessTimer:
    def __init__(self):
        self.last_time = None
        self.cur_time = None

    def start_time(self):
        """
        开始计时
        :return: 开始计时的时间
        """
        self.last_time =  time.time()
        self.cur_time = self.last_time

    # 结束计时
    def end_time(self, message):
        """
        结束计时
        :param message: 参与计时的函数功能
        :return: 空
        """
        self.cur_time = time.time()
        dur = self.cur_time - self.last_time
        print_warning(message + "：" + str(dur) + "s")
        self.last_time = self.cur_time


def cost_time(func):
    """
    用于计算耗时的装饰器
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        s_time = time.perf_counter()
        result = func(*args, **kwargs)
        e_time = time.perf_counter()
        print(f'func {func.__name__} cost time: {e_time - s_time:.8f}s')
        return result

    return wrapper
