from datetime import datetime
from functools import wraps


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        print(f"func: {f.__name__} took: {te-ts}")
        return result

    return wrap
