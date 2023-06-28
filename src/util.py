import math
import datetime

FPS = 60

def datetime2nframe(time: datetime.timedelta):
    seconds = time.seconds
    microseconds = time.microseconds
    sec = seconds + microseconds / 1000000
    nframe = sec * FPS
    return int(nframe)

time = datetime.timedelta(seconds=8, microseconds=67900)
nframe = datetime2nframe(time)
print(nframe)
