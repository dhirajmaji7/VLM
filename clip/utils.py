import time
import datetime
import functools

def timeit(func):
    """
    Decorator to time the execution of a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        duration_str = str(datetime.timedelta(seconds=int(duration)))
        print(f"[{func.__name__}] Execution time: {duration_str}")

        return result
    return wrapper
