
def get_data():
    basic_freq = int(input('Podaj częstotliwość sygnału analogowego:'))

    return basic_freq


def get_quantization_frequency():
    fs = int(input('Podaj częstotliwość kwantyzacji:'))

    return fs


def get_clip_value():
    clip_value = float(input('Where to clip:'))

    return clip_value


def get_n_sinc():
    n = int(input('Podaj liczbe próbek do granicy sumowania:'))

    return n

