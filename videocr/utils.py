import datetime

# convert time string to frame index
def get_frame_index(time_str: str, fps: float):
    t = time_str.split(':')
    t = list(map(float, t))
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(
            'Time data "{}" does not match format "%H:%M:%S"'.format(time_str))
    index = int(td.total_seconds() * fps)
    return index


# convert frame index into SRT/ASS timestamp
def get_timestamp(frame_index: int, fps: float, ass: bool):
    td = datetime.timedelta(seconds=frame_index / fps)
    ms = td.microseconds // 1000
    ms_ass = round(ms/10)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    if ass:
        return '{:01d}:{:02d}:{:02d}.{:02d}'.format(h, m, s, ms_ass)
    else:
        return '{:02d}:{:02d}:{:02d},{:03d}'.format(h, m, s, ms)
