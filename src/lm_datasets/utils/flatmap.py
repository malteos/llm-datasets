"""

Mostly based on MisterMiyagi's answer https://stackoverflow.com/a/71019318

"""

import functools
from multiprocess import Pool, Manager, Queue

# from multiprocessing import Pool, Manager, Queue


def flatmap(pool: Pool, func, iterable, chunksize=None):
    """A flattening, unordered equivalent of Pool.map()"""
    # use a queue to stream individual results from processes
    queue = Manager().Queue()
    # reuse task management and mapping of Pool
    pool.map_async(
        functools.partial(_flat_mappper, queue, func),
        iterable,
        chunksize,
        # callback: push a signal that everything is done
        lambda _: queue.put(None),
        lambda err: queue.put((None, err)),
    )
    # yield each result as it becomes available
    while True:
        item = queue.get()
        if item is None:
            break
        result, err = item
        if err is None:
            yield result
        else:
            raise err


def _flat_mappper(queue: Queue, func, *args):
    """Helper to run `(*args) -> iter` and stream results to a queue"""
    for item in func(*args):
        queue.put((item, None))
