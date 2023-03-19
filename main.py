import numpy as np
import matplotlib.pyplot as plt
import math

def get_input():
    time_start = int(input('Podaj czas początkowy:'))
    duration = int(input('Podaj czas trwania sygnału:'))
    time_to_end = time_start + duration
    amplitude = float(input('Podaj amplitude sygnału:'))
    sampling_rate = int(input('Podaj częstotliwość próbkowania:'))

    return time_start, time_to_end, amplitude, sampling_rate

def draw_graph(name, time_start, time_to_end, amplitude, sampling_rate, values_y):
    # Create a time axis for the signal
    t = np.linspace(time_start, time_to_end, sampling_rate)

    plt.plot(t, values_y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(name)
    plt.show()


def constant_noise():
    time_start, time_to_end, amplitude, sampling_rate = get_input()

    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    # Set the sampling rate and duration of the signal
    values_y = np.random.uniform(-amplitude/2, amplitude/2, nr_of_samplings)

    draw_graph("Constant noise", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    histogram(values_y)


def gaussian_noise():
    time_start, time_to_end, amplitude, sampling_rate = get_input()

    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    # Set the sampling rate and duration of the signal
    values_y = np.random.normal(0, amplitude/2, nr_of_samplings)

    draw_graph("Gaussian noise", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    histogram(values_y)

def sinus_signal():
    #time_start, time_to_end, amplitude, sampling_rate = get_input()
    time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    frequency = 2 * np.pi / basic_period
    #phase = np.pi / 4 # przesuniecie w fazie

    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    signal = amplitude * np.sin(frequency * time)
    draw_graph("sinus_signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    histogram(signal)

def sinus_half_straight_signal():
    #time_start, time_to_end, amplitude, sampling_rate = get_input()
    time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    frequency = 2 * np.pi / basic_period
    #phase = np.pi / 4 # przesuniecie w fazie

    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    signal = amplitude / 2 * (np.sin(frequency * time) + abs(np.sin(frequency * time)))
    draw_graph("sinus_half_straight_signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    histogram(signal)


def sinus_double_half_straight_signal():
    #time_start, time_to_end, amplitude, sampling_rate = get_input()
    time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    frequency = 2 * np.pi / basic_period
    #phase = np.pi / 4 # przesuniecie w fazie

    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    signal = amplitude * abs(np.sin(frequency * time))
    draw_graph("sinus_double_half_straight_signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    histogram(signal)


def rectangular_signal(): #6    #todo: period musi byc intem - kaszan, trzeba to zmienic
    #time_start, time_to_end, amplitude, sampling_rate = get_input()
    time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = int(input('Podaj okres podstawowy sygnału:'))
    fill_value = float(input('Podaj współczynnik wypełnienia sygnału:'))
    # basic_period = 2
    # fill_value = 0.5

    nr_of_samplings = sampling_rate * (time_to_end - time_start)
    values_y = np.zeros(nr_of_samplings)

    for x in range(time_start, time_to_end, basic_period):
        for i in range(int(basic_period * fill_value * sampling_rate)):
            values_y[x * sampling_rate + i] = amplitude

    draw_graph("Rectangular signal", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    histogram(values_y)
#---------------------------------------------------------------------
    #time_start, time_to_end, amplitude, sampling_rate = get_input()
    # time_start, time_to_end, amplitude, sampling_rate = -5, 10, 10, 100
    # basic_period = float(input('Podaj okres podstawowy sygnału:'))
    # fill_value = float(input('Podaj współczynnik wypełnienia sygnału:'))
    # nr_of_samplings = sampling_rate * (time_to_end - time_start)
    # values_y = np.zeros(nr_of_samplings)
    #
    # frequency = 2 * np.pi / basic_period
    # #phase = np.pi / 4 # przesuniecie w fazie
    #
    # time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end
    #
    # # if time <
    # #     signal = amplitude * abs(np.sin(frequency * time))
    #
    # for x in range(time_start, time_to_end, int(basic_period)):
    #     for i in range(int(basic_period * fill_value * sampling_rate)):
    #         values_y[x * sampling_rate + i] = amplitude
    #
    # draw_graph("rectangular_signal", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    # histogram(values_y)

    # Define the parameters of the rectangular signal

def rectangular_symmetrical_signal(): #7
    #time_start, time_to_end, amplitude, sampling_rate = get_input()
    time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = int(input('Podaj okres podstawowy sygnału:'))
    fill_value = float(input('Podaj współczynnik wypełnienia sygnału:'))
    # basic_period = 2
    # fill_value = 0.5

    nr_of_samplings = sampling_rate * (time_to_end - time_start)
    values_y = np.zeros(nr_of_samplings)
    for x in range(nr_of_samplings):
        values_y[x] = -amplitude

    for x in range(time_start, time_to_end, basic_period):
        for i in range(int(basic_period * fill_value * sampling_rate)):
            values_y[x * sampling_rate + i] = amplitude

    # for x in range(time_start * sampling_rate, time_to_end * sampling_rate, basic_period * sampling_rate):
    #     for i in range(int(basic_period * fill_value * sampling_rate)):
    #         values_y[x + i] = amplitude

    draw_graph("Rectangular symmetrical signal", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    histogram(values_y)

def interpolate(x1: float, x2: float, y1: float, y2: float, x: float):
    """Perform linear interpolation for x between (x1,y1) and (x2,y2) """

    return ((y2 - y1) * x + x2 * y1 - x1 * y2) / (x2 - x1)

# def triangular_signal(): #8
#     # time_start, time_to_end, amplitude, sampling_rate = get_input()
#     time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
#     # basic_period = float(input('Podaj okres podstawowy sygnału:'))
#     # fill_value = float(input('Podaj współczynnik wypełnienia sygnału:'))
#     basic_period = 2
#     fill_value = 0.5
#
#     nr_of_samplings = sampling_rate * (time_to_end - time_start)
#     values_y = np.zeros(nr_of_samplings)
#
#     for x in range(time_start, time_to_end, basic_period):
#         for i in range(int(basic_period * fill_value * sampling_rate)):
#             values_y[x * sampling_rate + i] = interpolate(x * sampling_rate,
#                                                           x * sampling_rate + basic_period * fill_value,
#                                                           0,
#                                                           amplitude,
#                                                           x * sampling_rate + i)
#
#     draw_graph("Triangular signal", time_start, time_to_end, amplitude, nr_of_samplings, values_y)

# def triangular_signal(): #8
#     #time_start, time_to_end, amplitude, sampling_rate = get_input()
#     time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 10
#     fill_value = float(input('Podaj współczynnik wypełnienia sygnału:'))
#     period = int(input('Podaj okres: '))
#
#     nr_of_samplings = sampling_rate * (time_to_end - time_start)
#     values_y = np.zeros(nr_of_samplings)
#
#     for y in range(time_start, time_to_end):    # dla kazdej sekundy
#         for x in range(0, sampling_rate):   # dla kazdego sampla w danej sekundzie
#             k = y # y * sampling_rate   # zakladajac ze y jest calkowite
#             t = y * sampling_rate + x
#             # if time_start < 0:
#             #     if t > ts * sampling_rate:
#             #         values_y[t - time_start * sampling_rate] = amplitude
#             #     elif t == ts * sampling_rate:
#             #         values_y[t - time_start * sampling_rate] = amplitude / 2
#             #     else:
#             #         values_y[t - time_start * sampling_rate] = 0
#             # if time_start >= 0:
#
#             #parzyste
#             if t < k*period* sampling_rate + time_start* sampling_rate or t < fill_value * period* sampling_rate + k*period* sampling_rate + time_start* sampling_rate:
#                 values_y[t] = amplitude * (t - k*period* sampling_rate - time_start* sampling_rate) / (fill_value * period* sampling_rate)
#                 print('es1')
#             elif t < fill_value * period* sampling_rate + time_start* sampling_rate + k*period* sampling_rate or t < period* sampling_rate + k*period* sampling_rate + time_start* sampling_rate:
#                 values_y[t] = - amplitude * (t - k*period* sampling_rate - time_start* sampling_rate) / (period* sampling_rate * (1 - fill_value)) + amplitude / (1 - fill_value)
#                 print('es2')
#             # #nieparzyste
#             # elif t < (2 * k + 1) * period* sampling_rate + time_start* sampling_rate or t < fill_value * period* sampling_rate + (2 * k + 1) * period* sampling_rate + time_start* sampling_rate:
#             #     values_y[t] = amplitude * (t - (2 * k + 1) * period* sampling_rate - time_start* sampling_rate) / (fill_value * period* sampling_rate)
#             #     print('es3')
#             # elif t < fill_value * period* sampling_rate + time_start* sampling_rate + (2 * k + 1) * period* sampling_rate or t < period* sampling_rate + (2 * k + 1) * period* sampling_rate + time_start* sampling_rate:
#             #     values_y[t] = - amplitude * (t - (2 * k + 1) * period* sampling_rate - time_start* sampling_rate) / (
#             #                 period * sampling_rate* (1 - fill_value)) + amplitude / (1 - fill_value)
#             #     print('es4')
#             else:
#                 print('dupa')
#
#     draw_graph("Triangular signal", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
#     histogram(values_y)

def triangular_signal(): #8
    time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)
    # Define the parameters of the triangular signal
    frequency = 2 * np.pi / basic_period
    duration = time_to_end - time_start

    # Create the time axis
    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    # Create the triangular signal
    signal = amplitude * np.abs(2 * (time * frequency - np.floor(time * frequency + 0.5)))
    # draw_graph("Triangular signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    # Plot the signal
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Triangular Signal')
    plt.show()
    histogram(signal)

def jump_signal(): #9
    # time_start, time_to_end, amplitude, sampling_rate = get_input()
    time_start, time_to_end, amplitude, sampling_rate = -10, 10, 10, 10
    ts = int(input('Podaj czas skoku:'))

    nr_of_samplings = sampling_rate * (time_to_end - time_start)
    values_y = np.zeros(nr_of_samplings)

    for y in range(time_start, time_to_end):
        for x in range(0, sampling_rate):
            t = y * sampling_rate + x
            if t > ts * sampling_rate:
                values_y[t - time_start * sampling_rate] = amplitude
            elif t == ts * sampling_rate:
                values_y[t - time_start * sampling_rate] = amplitude / 2
            else:
                values_y[t - time_start * sampling_rate] = 0

    draw_graph("Jump signal", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    histogram(values_y)


def unitary_impuls():
    # # time_start, time_to_end, amplitude, sampling_rate = get_input()
    # time_start, time_to_end, amplitude, sampling_rate = -2, 2, 10, 50
    # #ts = int(input('Podaj czas skoku:'))
    # nr_of_samplings = sampling_rate * (time_to_end - time_start)
    # #duration = time_to_end - time_start
    # location = 1
    #
    # # Generate the time axis
    # time = np.arange(time_start, time_to_end, time_to_end / sampling_rate)
    #
    # # Generate the impulse signal
    # signal = np.zeros(len(time))
    # signal[int(location * sampling_rate)] = amplitude
    #
    # # Normalize the signal
    # signal = signal / np.sqrt(np.mean(signal ** 2))
    #
    # plt.scatter(time, signal)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.show()


    # time_start, time_to_end, amplitude, sampling_rate = get_input()
    time_start, time_to_end, amplitude, sampling_rate = -10, 10, 10, 2
    #ts = int(input('Podaj czas skoku:'))

    nr_of_samplings = sampling_rate * (time_to_end - time_start)
    values_y = np.zeros(nr_of_samplings)
    time = np.arange(time_start * sampling_rate, time_to_end * sampling_rate, 1)

    for y in range(time_start, time_to_end):
        for x in range(0, sampling_rate):
            t = y * sampling_rate + x
            if t == 0:
                values_y[t - time_start * sampling_rate] = 1
            else:
                values_y[t - time_start * sampling_rate] = 0

    #draw_graph("Jump signal", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    plt.scatter(time, values_y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    histogram(values_y)


def noise_impuls():
    return 0

def histogram(seq) -> dict:
     hist = {}
     for i in seq:
         hist[i] = hist.get(i, 0) + 1
     plt.hist(hist)
     plt.show()
     return 0

# main:
user_input = int(input('1 sinus\n2 gausian\n3 constant\n4 sinus half straight\n5 sinus double half straight\n6 rectangular'
                       '\n7 rectangular symmetrical\n8 triangular\n9 jump\n10 unitary impuls\n11 noise impuls'))
if user_input == 1:
    sinus_signal()
elif user_input == 2:
    gaussian_noise()
elif user_input == 3:
    constant_noise()
elif user_input == 4:
    sinus_half_straight_signal()
elif user_input == 5:
    sinus_double_half_straight_signal()
elif user_input == 6:
    rectangular_signal()
elif user_input == 7:
    rectangular_symmetrical_signal()
elif user_input == 8:
    triangular_signal()
elif user_input == 9:
    jump_signal()
elif user_input == 10:
    unitary_impuls()
elif user_input == 11:
    noise_impuls()
else:
    print('niewlasciwy input')