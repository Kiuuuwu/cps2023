import copy

import user_communication
import main
import numpy as np
import matplotlib.pyplot as plt
import math

def constant_sampling(time_start, time_end, if_draw=False):

    basic_freq = user_communication.get_data()
    original_signal, start, end = main.sinus_signal(basic_freq, time_start, time_end, False)
    fs = user_communication.get_quantization_frequency()
    fs = int(basic_freq / fs)
    signal_sampled = original_signal[::fs]
    if if_draw:
        samples_num = len(signal_sampled)
        main.draw_graph("Constant sampling", time_start, time_end, basic_freq * (time_end - time_start) / fs, samples_num, signal_sampled)

    return signal_sampled, basic_freq, original_signal


def constant_quantization_with_clip(time_start, time_end, if_draw=True):
    basic_freq = user_communication.get_data()
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

    return new_signal, basic_freq, smooth_signal


def constant_quantization_with_rounding(time_start, time_end, if_draw):
    basic_freq = user_communication.get_data()
    input_signal, start, end = main.sinus_signal(basic_freq, time_start, time_end, False)
    clip_value = user_communication.get_clip_value()
    new_signal = copy.deepcopy(input_signal)
    for x in range(len(input_signal)):
        new_signal[x] = round(input_signal[x], clip_value)

    t = np.linspace(int(time_start), int(time_end), int(basic_freq * (time_end - time_start)))
    if if_draw:
        fig, ax = plt.subplots()
        ax.plot(t, input_signal, label="original")
        ax.plot(t, new_signal, label="quantizied")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        plt.show()

    return new_signal, basic_freq, input_signal

def zero_order_hold_reconstruction(time_start, time_end):
    # signal, signal_freq, original_signal = constant_quantization_with_clip(time_start, time_end, if_draw=False)
    signal, signal_freq, original_signal = constant_quantization_with_rounding(time_start, time_end, if_draw=False)
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
    # ax.plot(t1, signal, label="original")
    ax.plot(t2, new_signal, label="reconstructed ZOH")
    ax.plot(t1, original_signal, label="original sin")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()

    return new_signal, original_signal


def first_order_hold_reconstruction(time_start, time_end):
    # signal, original_freq, original_signal = constant_quantization_with_clip(time_start, time_end, if_draw=False)
    signal, original_freq, original_signal = constant_quantization_with_rounding(time_start, time_end, if_draw=False)
    # DLA CONSTANT SAMPLING NIE DZIALA, POZA TYM OK

    reconstructed = []
    foh_time = []
    reconstructed.append(signal[0])
    foh_time.append(time_start)
    for x in range(1, len(signal)):
        if signal[x] != reconstructed[-1]:
            reconstructed.append(signal[x])
            foh_time.append(time_start + x * (1/original_freq))

    reconstructed = np.array(reconstructed)
    foh_time = np.array(foh_time)

    t1 = np.linspace(int(time_start), int(time_end), int(original_freq * (time_end - time_start)))
    fig, ax = plt.subplots()
    ax.plot(t1, original_signal, label="original")
    ax.plot(foh_time, reconstructed, label="reconstructed FOH")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()

    return reconstructed, original_signal


def sinc_function(x):
    if x == 0:
        output = 1
    else:
        output = math.sin(np.pi * x) / (np.pi * x)

    return output


def sinc_reconstruction(time_start, time_end):
    #todo: jak jest za mało próbek, to wykres robi się zero
    N = user_communication.get_n_sinc()
    signal, original_freq, original_signal = constant_quantization_with_clip(time_start, time_end, if_draw=False)
    # signal, original_freq, original_signal = constant_quantization_with_rounding(time_start, time_end, if_draw= False)
    # DLA CONSTANT SAMPLING NIE DZIALA, POZA TYM OK
    # signal, original_freq, original_signal = constant_sampling(time_start, time_end, if_draw=False)
    ts = 1 / original_freq
    reconstructed = np.zeros(len(signal))
    for i in range(len(signal)):
        t = i * ts
        for n in range(-N, N):
            # reconstructed[i] += signal[i] * n * ts * sinc_function((t / ts) - n)
            reconstructed[i] += signal[i] * sinc_function((t / ts) - n)


    t1 = np.linspace(int(time_start), int(time_end), int(original_freq * (time_end - time_start)))
    fig, ax = plt.subplots()
    ax.plot(t1, signal, label="digital")
    ax.plot(t1, reconstructed, label="reconstructed")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()

    return reconstructed, original_signal


def mean_squared_error(signal1, signal2):
    if len(signal1) != len(signal2):
        print("Liczby probek obu sygnalow nie sa rowne")
        return 0

    n = len(signal1)
    mse = 0
    for i in range(n-1):
        mse += pow(signal1[i] - signal2[i], 2)

    mse /= n
    return mse


def peak_signal_to_noise_ratio(original_signal, reconstructed_signal):
    numerator = original_signal[0]
    for i in range(1, len(original_signal)):
        if original_signal[i] > numerator:
            numerator = original_signal[i]

    denominator = signal_to_noise_ratio(original_signal, reconstructed_signal)
    if denominator == 0:
        print("Nie dzielic przez zero, cholero")
        return 0

    psnr = 10 * math.log(numerator/denominator, 10)
    return psnr


def signal_to_noise_ratio(original_signal, reconstructed_signal):
    if len(original_signal) != len(reconstructed_signal):
        print("Liczby probek obu sygnalow nie sa rowne")
        return 0

    n = len(original_signal)
    snr = 0
    numerator = 0
    denominator = 0
    for i in range(n-1):
        numerator += pow(original_signal[i], 2)
        denominator += pow(original_signal[i] - reconstructed_signal[i], 2)

    if denominator == 0:
        print("Nie dzielic przez zero, cholero")
        return 0

    snr = 10 * math.log(numerator/denominator, 10)
    return snr


def maximum_difference(original_signal, reconstructed_signal):
    if len(original_signal) != len(reconstructed_signal):
        print("Liczby probek obu sygnalow nie sa rowne")
        return 0

    md = 0
    for i in range(len(original_signal)):
        current_difference = abs(original_signal[i] - reconstructed_signal[i])
        if current_difference > md:
            md = current_difference

    return md


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


# signal, time_start, time_to_end = main.sinus_signal(1000, 0, 6, False)

time_start = 0
time_end = 2

# reconstructed = sinc_reconstruction(time_start, time_end)
reconstructed, original = zero_order_hold_reconstruction(time_start, time_end)


print("MSE:  ", mean_squared_error(reconstructed, original))
print("SNR: ", signal_to_noise_ratio(reconstructed, original))
print("PSNR: ", peak_signal_to_noise_ratio(reconstructed, original))
print("MD: ", maximum_difference(reconstructed, original))

# first_order_hold_reconstruction()
# constant_quantization_with_rounding(True)
# constant_quantization_with_clip()
# constant_sampling()
# test()
# sinc_reconstruction()
# zero_order_hold_reconstruction()