
def get_data():
    time_start = float(input('Podaj czas początkowy:'))
    time_end = float(input('Podaj czas koncowy:'))
    basic_freq = int(input('Podaj częstotliwość sygnału analogowego:'))

    return time_start, time_end, basic_freq


def get_quantization_frequency():
    fs = int(input('Podaj częstotliwość kwantyzacji:'))

    return fs


def get_clip_value():
    clip_value = float(input('Where to clip:'))

    return clip_value

