import copy

import user_communication
import main
import numpy as np
import matplotlib.pyplot as plt

def constant_sampling():

    time_start, time_end, basic_freq = user_communication.get_data()
    signal, start, end = main.sinus_signal(basic_freq, time_start, time_end, False)
    fs = user_communication.get_quantization_frequency()
    fs = int(basic_freq / fs)
    signal_sampled = signal[::fs]
    samples_num = len(signal_sampled)
    main.draw_graph("Constant sampling", time_start, time_end, basic_freq * (time_end - time_start) / fs, samples_num, signal_sampled)


def constant_quantization_with_clip(if_draw=True):
    time_start, time_end, basic_freq = user_communication.get_data()
    smooth_signal, start, end = main.sinus_signal(basic_freq, time_start, time_end, False)
    clip_value = user_communication.get_clip_value()
    new_signal = copy.deepcopy(smooth_signal)
    for x in range(len(smooth_signal)):
        original_value = smooth_signal[x]
        v = original_value % clip_value
        v1 = original_value - v
        new_signal[x] = v1

    if if_draw:
        t = np.linspace(int(time_start), int(time_end), int(basic_freq * (time_end - time_start)))
        fig, ax = plt.subplots()
        ax.plot(t, smooth_signal, label="original")
        ax.plot(t, new_signal, label="quantizied")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        plt.show()

    return time_start, time_end, new_signal, basic_freq, smooth_signal


def constant_quantization_with_rounding(if_draw):
    time_start, time_end, basic_freq = user_communication.get_data();
    signal, start, end = main.sinus_signal(basic_freq, time_start, time_end, False)
    clip_value = user_communication.get_clip_value()
    new_signal = copy.deepcopy(signal)
    for x in range(len(signal)):
        new_signal[x] = round(signal[x], clip_value)

    t = np.linspace(int(time_start), int(time_end), int(basic_freq * (time_end - time_start)))
    if if_draw:
        fig, ax = plt.subplots()
        ax.plot(t, signal, label="original")
        ax.plot(t, new_signal, label="quantizied")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        plt.show()

    return time_start, time_end, new_signal, basic_freq

def zero_order_hold_reconstruction():
    #todo: ustalic czy dobrze, czy schodki powinny byc przesuniete
    time_start, time_end, signal, signal_freq, smooth_signal = constant_quantization_with_clip(if_draw=False)
    frequency = 1000
    new_signal = np.zeros(frequency * int(time_end - time_start))
    original_samples_num = len(signal)
    new_samples_num = len(new_signal)

    ratio = int(new_samples_num / original_samples_num)
    for x in range(original_samples_num):
        for y in range(ratio):
            new_signal[x * ratio + y] = copy.deepcopy(signal[x])

    t1 = np.linspace(int(time_start), int(time_end), int(signal_freq * (time_end - time_start)))
    t2 = np.linspace(int(time_start), int(time_end), int(frequency * (time_end - time_start)))
    fig, ax = plt.subplots()
    ax.plot(t1, signal, label="original")
    ax.plot(t2, new_signal, label="reconstructed")
    ax.plot(t1, smooth_signal, label="smooth sin")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()


def first_order_hold_reconstruction():
    time_start, time_end, signal, original_freq, smooth_signal = constant_quantization_with_clip(if_draw=False)
    first_order_array = []
    foh_time = []
    first_order_array.append(signal[0])
    foh_time.append(time_start)
    for x in range(1, len(signal)):
        if signal[x] != first_order_array[-1]:
            first_order_array.append(signal[x])
            foh_time.append(time_start + x * (1/original_freq))

    first_order_array = np.array(first_order_array)
    # foh_time += time_start
    foh_time = np.array(foh_time)
    print(foh_time)
    # plt.scatter(foh_time, first_order_array)

    t1 = np.linspace(int(time_start), int(time_end), int(original_freq * (time_end - time_start)))
    t2 = np.linspace(int(time_start), int(time_end), len(first_order_array))
    fig, ax = plt.subplots()
    ax.plot(t1, signal, label="digital")
    ax.plot(foh_time, first_order_array, label="reconstructed")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()


def test(): # mozliwe ze przy ujemnych dla wartosci srodkowej zaokragla sie w zla strone, ale nie wiemy
    original_value = -13.75
    clip_value = 0.5
    v = original_value % clip_value
    lower = original_value - v
    upper = lower + clip_value
    if abs(original_value - lower) < abs(original_value - upper):
        rounded = lower
    else:
        rounded = upper

    if abs(original_value - lower) == abs(original_value - upper):
        if original_value < 0:
            rounded = lower
        else:
            rounded = upper

    print(rounded)


def round(original_value, clip_value):
    v = original_value % clip_value
    lower = original_value - v
    upper = lower + clip_value
    if abs(original_value - lower) < abs(original_value - upper):
        rounded = lower
    else:
        rounded = upper

    if abs(original_value - lower) == abs(original_value - upper):
        if original_value < 0:
            rounded = lower
        else:
            rounded = upper

    return rounded

# test()
# zero_order_hold_reconstruction()
first_order_hold_reconstruction()
# constant_quantization_with_rounding(True)
# constant_quantization_with_clip()
# constant_sampling()