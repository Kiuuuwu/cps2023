import copy

import main
import numpy as np
import matplotlib.pyplot as plt

def constant_sampling():
    time_start = float(input('Podaj czas początkowy:'))
    time_end = float(input('Podaj czas koncowy:'))
    basic_freq = int(input('Podaj częstotliwość sygnału analogowego:'))
    signal, start, end = main.sinus_signal(basic_freq, time_start, time_end)
    fs = int(input('Podaj częstotliwość kwantyzacji:')) # częstotliwość kwantyzacji
    fs = int(basic_freq / fs)
    signal_sampled = signal[::fs]
    samples_num = len(signal_sampled)
    main.draw_graph("Constant sampling", time_start, time_end, basic_freq * (time_end - time_start) / fs, samples_num, signal_sampled)


def constant_quantization_with_clip():
    time_start = float(input('Podaj czas początkowy:'))
    time_end = float(input('Podaj czas koncowy:'))
    basic_freq = int(input('Podaj częstotliwość sygnału analogowego:'))
    signal, start, end = main.sinus_signal(basic_freq, time_start, time_end)
    clip_value = float(input('Where to clip:'))
    new_signal = copy.deepcopy(signal)
    for x in range(len(signal)):
        original_value = signal[x]
        v = original_value % clip_value
        v1 = int(original_value - v)
        new_signal[x] = v1

    t = np.linspace(int(time_start), int(time_end), int(basic_freq * (time_end - time_start)))
    fig, ax = plt.subplots()
    ax.plot(t, signal, label="original")
    ax.plot(t, new_signal, label="quantizied")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    # ax.legend()
    plt.show()


constant_quantization_with_clip()
# constant_sampling()