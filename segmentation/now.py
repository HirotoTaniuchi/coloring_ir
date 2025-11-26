import datetime
import os

def now1():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")

def now2():
    return datetime.datetime.now().strftime("%Y%m%d")

if __name__ == "__main__":
    print(now1())
    print(now2())
