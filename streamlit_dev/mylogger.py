import time

def log(
    message: str,
):
    time_ = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{time_}] LOG: {message}")