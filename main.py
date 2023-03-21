import csv

import numpy as np
import matplotlib.pyplot as plt
import math

def count_means(signal):
    mean = np.mean(signal)
    abs_mean = np.mean(np.abs(signal))
    effective_value = np.sqrt(np.mean(signal ** 2))
    variance = np.var(signal)
    med_power = np.median(signal ** 2)

    print("Mean value:", mean)
    print("Absolute mean value:", abs_mean)
    print("Effective value (RMS):", effective_value)
    print("Variance:", variance)
    print("Median power:", med_power)
def get_input(sampling_rate, time_start, time_to_end):
    if time_start == -1001:
        time_start = int(input('Podaj czas początkowy:'))
    if time_to_end == -1001:
        duration = int(input('Podaj czas trwania sygnału:'))
        time_to_end = time_start + duration
    amplitude = float(input('Podaj amplitude sygnału:'))
    if sampling_rate == 0:
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


def constant_noise(sampling_rate1, time_start1, time_to_stop1):
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)

    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    # Set the sampling rate and duration of the signal
    values_y = np.random.uniform(-amplitude/2, amplitude/2, nr_of_samplings)

    draw_graph("Constant noise", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    histogram(values_y)

    return values_y, time_start, time_to_end


def gaussian_noise(sampling_rate1, time_start1, time_to_stop1):
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)

    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    # Set the sampling rate and duration of the signal
    values_y = np.random.normal(0, amplitude/2, nr_of_samplings)

    draw_graph("Gaussian noise", time_start, time_to_end, amplitude, nr_of_samplings, values_y)
    histogram(values_y)

    return values_y, time_start, time_to_end

def sinus_signal(sampling_rate1, time_start1, time_to_stop1):
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    #time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    frequency = 2 * np.pi / basic_period

    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    signal = amplitude * np.sin(frequency * time)
    draw_graph("sinus_signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    histogram(signal)

    count_means(signal)

    return signal, time_start, time_to_end


def sinus_half_straight_signal(sampling_rate1, time_start1, time_to_stop1):
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    #time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    frequency = 2 * np.pi / basic_period

    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    signal = amplitude / 2 * (np.sin(frequency * time) + abs(np.sin(frequency * time)))
    draw_graph("sinus_half_straight_signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    histogram(signal)

    count_means(signal)

    return signal, time_start, time_to_end


def sinus_double_half_straight_signal(sampling_rate1, time_start1, time_to_stop1):
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    #time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    frequency = 2 * np.pi / basic_period

    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    signal = amplitude * abs(np.sin(frequency * time))
    draw_graph("sinus_double_half_straight_signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    histogram(signal)

    count_means(signal)

    return signal, time_start, time_to_end

def rectangular_signal(sampling_rate1, time_start1, time_to_stop1): #6
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    basic_period = int(input('Podaj okres podstawowy sygnału:'))
    frequency = 1 / basic_period
    fill_value = float(input('Podaj współczynnik sygnału:'))
    nr_of_samples = (time_to_end - time_start) * sampling_rate
    time = np.arange(time_start, time_to_end, 1 / sampling_rate)

    values_y = np.where(np.mod(time, 1 / frequency) <  fill_value / frequency, amplitude, 0)
    # Plot the rectangular signal
    draw_graph("Rectangular  signal", time_start, time_to_end, amplitude, nr_of_samples, values_y)

    histogram(values_y)

    count_means(values_y)
    return values_y, time_start, time_to_end


def rectangular_symmetrical_signal(sampling_rate1, time_start1, time_to_stop1): #7
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    basic_period = int(input('Podaj okres podstawowy sygnału:'))
    frequency = 1/ basic_period # in Hz
    fill_value = float(input('Podaj współczynnik sygnału:'))
    nr_of_samples = (time_to_end - time_start) * sampling_rate
    time = np.arange(time_start, time_to_end, 1 / sampling_rate)

    values_y = np.where(np.mod(time, 1 / frequency) < fill_value / frequency, amplitude, -amplitude)
    # Plot the rectangular signal
    draw_graph("Rectangular symmetrical signal", time_start, time_to_end, amplitude, nr_of_samples, values_y)

    histogram(values_y)

    count_means(values_y)
    return values_y, time_start, time_to_end


def interpolate(x1: float, x2: float, y1: float, y2: float, x: float):
    """Perform linear interpolation for x between (x1,y1) and (x2,y2) """

    return ((y2 - y1) * x + x2 * y1 - x1 * y2) / (x2 - x1)

def triangular_signal(sampling_rate1, time_start1, time_to_stop1): #8
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    #time_start, time_to_end, amplitude, sampling_rate = 0, 10, 10, 100
    basic_period = float(input('Podaj okres podstawowy sygnału:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)
    frequency = 2 * np.pi / basic_period

    time = np.linspace(time_start, time_to_end, nr_of_samplings)  # nr_of_samplings samples between time_start and time_to_end

    signal = amplitude * np.abs(2 * (time * frequency - np.floor(time * frequency + 0.5)))
    draw_graph("Triangular signal", time_start, time_to_end, amplitude, nr_of_samplings, signal)
    histogram(signal)

    count_means(signal)

    return signal, time_start, time_to_end

def jump_signal(sampling_rate1, time_start1, time_to_stop1): #9
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    #time_start, time_to_end, amplitude, sampling_rate = -10, 10, 10, 50
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

    count_means(values_y)

    return values_y, time_start, time_to_end


def unitary_impuls(sampling_rate1, time_start1, time_to_stop1):
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    #time_start, time_to_end, amplitude, sampling_rate = -10, 10, 10, 50
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

    plt.scatter(time, values_y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    histogram(values_y)

    return values_y, time_start, time_to_end


def noise_impuls(sampling_rate1, time_start1, time_to_stop1):
    time_start, time_to_end, amplitude, sampling_rate = get_input(sampling_rate1, time_start1, time_to_stop1)
    #time_start, time_to_end, amplitude, sampling_rate = -10, 10, 10, 50
    possibility = float(input('Podaj prawdopodobienstwo:'))
    nr_of_samplings = sampling_rate * (time_to_end - time_start)

    time = np.arange(time_start * sampling_rate, time_to_end * sampling_rate, 1)
    signal_length = nr_of_samplings
    noise = np.zeros(len(time))

    impulse = np.random.rand(signal_length) < possibility
    noise[impulse] = 1

    plt.scatter(time, noise)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    histogram(noise)

    return noise, time_start, time_to_end

def histogram(seq) -> dict:
     hist = {}
     for i in seq:
         hist[i] = hist.get(i, 0) + 1
     plt.hist(hist)
     plt.show()

def sum(signal1, t01, tk1, signal2, t02, tk2, sampling_rate):
    # Define time vector
    #t = np.linspace(0, 2 * np.pi, 1000) #niech to bedzie czas startowy wczesniejszego syganlu i czas koncowy ponzniejszego,

    time_start = t01
    time_to_end = tk1

    if t01 > t02:
        time_start = t02
    if tk1 < tk2:
        time_to_end = tk2

    t = np.linspace(time_start, time_to_end, sampling_rate * (time_to_end - time_start))  #zeby to dzialalo to trzebaby miec ujednolicone probkowanie dla obu dodawanych sygnalow

    #mozna zrobic filowanie zerami dwa sygnaly zeby tych zer bylo tyle ile sampling_rate * (time_to_end - time_start)
    # s1 = np.zeros(sampling_rate * (time_to_end - time_start))
    # s2 = np.zeros(sampling_rate * (time_to_end - time_start))
    #
    # # signal1 = s1 + signal1  #nie moze ich dodac jak maja rozna liczbe danych
    # # signal2 = s2 + signal2
    #
    # for x in s1:
    #     for y in signal1:



    signal_sum = signal1 + signal2

    fig, ax = plt.subplots()
    ax.plot(t, signal1, label=signal1)
    ax.plot(t, signal2, label=signal2)
    ax.plot(t, signal_sum, label='sum')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    #ax.legend()
    plt.show()

    return signal_sum

def subtract(signal1, t01, tk1, signal2, t02, tk2, sampling_rate):
    # Define time vector
    #t = np.linspace(0, 2 * np.pi, 1000) #niech to bedzie czas startowy wczesniejszego syganlu i czas koncowy ponzniejszego,

    time_start = t01
    time_to_end = tk1

    if t01 > t02:
        time_start = t02
    if tk1 < tk2:
        time_to_end = tk2

    t = np.linspace(time_start, time_to_end, sampling_rate * (time_to_end - time_start))  #zeby to dzialalo to trzebaby miec ujednolicone probkowanie dla obu dodawanych sygnalow

    signal_sum = signal1 - signal2

    fig, ax = plt.subplots()
    ax.plot(t, signal1, label=signal1)
    ax.plot(t, signal2, label=signal2)
    ax.plot(t, signal_sum, label='sum')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    #ax.legend()
    plt.show()

    return signal_sum

#todo: to nie wyglada na poprawne mnozenie
def multiply(signal1, t01, tk1, signal2, t02, tk2, sampling_rate):
    # Define time vector
    #t = np.linspace(0, 2 * np.pi, 1000) #niech to bedzie czas startowy wczesniejszego syganlu i czas koncowy ponzniejszego,

    time_start = t01
    time_to_end = tk1

    if t01 > t02:
        time_start = t02
    if tk1 < tk2:
        time_to_end = tk2

    t = np.linspace(time_start, time_to_end, sampling_rate * (time_to_end - time_start))  #zeby to dzialalo to trzebaby miec ujednolicone probkowanie dla obu dodawanych sygnalow
    #signal2[signal2 < 0] = 0
    signal_sum = signal1 * signal2

    fig, ax = plt.subplots()
    ax.plot(t, signal1, label=signal1)
    ax.plot(t, signal2, label=signal2)
    ax.plot(t, signal_sum, label='sum')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    #ax.legend()
    plt.show()

    return signal_sum

def divide(signal1, t01, tk1, signal2, t02, tk2, sampling_rate):
    # Define time vector
    #t = np.linspace(0, 2 * np.pi, 1000) #niech to bedzie czas startowy wczesniejszego syganlu i czas koncowy ponzniejszego,

    time_start = t01
    time_to_end = tk1

    if t01 > t02:
        time_start = t02
    if tk1 < tk2:
        time_to_end = tk2

    t = np.linspace(time_start, time_to_end, sampling_rate * (time_to_end - time_start))  #zeby to dzialalo to trzebaby miec ujednolicone probkowanie dla obu dodawanych sygnalow

    for i in range (0, signal2.size):
        if signal2[i] == 0:
            signal2[i] = 0.00001


    signal_sum = signal1 / signal2

    fig, ax = plt.subplots()
    ax.plot(t, signal1, label=signal1)
    ax.plot(t, signal2, label=signal2)
    ax.plot(t, signal_sum, label='sum')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    #ax.legend()
    plt.show()

    return signal_sum

def big_input(sampling_rate, time_start, time_to_end):
    user_input = int(
        input('1 sinus\n2 gausian\n3 constant\n4 sinus half straight\n5 sinus double half straight\n6 rectangular'
              '\n7 rectangular symmetrical\n8 triangular\n9 jump\n10 unitary impuls\n11 noise impuls'))
    signal = 0
    if user_input == 1:
        signal, time_start, time_to_end = sinus_signal(sampling_rate, time_start, time_to_end)
    elif user_input == 2:
        signal, time_start, time_to_end = gaussian_noise(sampling_rate, time_start, time_to_end)
    elif user_input == 3:
        signal, time_start, time_to_end = constant_noise(sampling_rate, time_start, time_to_end)
    elif user_input == 4:
        signal, time_start, time_to_end = sinus_half_straight_signal(sampling_rate, time_start, time_to_end)
    elif user_input == 5:
        signal, time_start, time_to_end = sinus_double_half_straight_signal(sampling_rate, time_start, time_to_end)
    elif user_input == 6:
        signal, time_start, time_to_end = rectangular_signal(sampling_rate, time_start, time_to_end)
    elif user_input == 7:
        signal, time_start, time_to_end = rectangular_symmetrical_signal(sampling_rate, time_start, time_to_end)
    elif user_input == 8:
        signal, time_start, time_to_end = triangular_signal(sampling_rate, time_start, time_to_end)
    elif user_input == 9:
        signal, time_start, time_to_end = jump_signal(sampling_rate, time_start, time_to_end)
    elif user_input == 10:
        signal, time_start, time_to_end = unitary_impuls(sampling_rate, time_start, time_to_end)
    elif user_input == 11:
        signal, time_start, time_to_end = noise_impuls(sampling_rate, time_start, time_to_end)
    else:
        print('niewlasciwy input')
    return signal, time_start, time_to_end


def save_to_csv(filename, time_start, time_to_end, t_step, signal):
    # Define the time axis
    t = np.arange(time_start * sampling_rate, time_to_end * sampling_rate, 1)
    # Calculate the sinusoidal signal
    #sin_signal = np.sin(t)

    # Save the signal to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Signal'])
        for i in range(len(t)):
            writer.writerow([t[i], signal[i]])

def read_from_csv(filename):
    # Read the signal data from the CSV file
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the header row
        t = []
        signal = []
        for row in reader:
            t.append(float(row[0]))
            signal.append(float(row[1]))

    return t, signal


# main:
filename = 'E:\politechnika\semestr6_lol_jeszcze_zyje\przetwarzanie_sygnalow\dane.csv'
user_input1 = int(input('1 signal or noise\n2 add\n3 subtract\n4 multiply\n5 divide\n6 read from file'))
if user_input1 == 1:
    sampling_rate = 0
    time_start = -1001
    time_to_stop = -1001
    signal = big_input(sampling_rate, time_start, time_to_stop)

elif user_input1 == 2:
    sampling_rate = int(input('enter sampling rate: '))
    time_start = int(input('enter starting time: '))
    time_to_stop = int(input('enter end time: '))

    print('choose what are you going to add')
    signal1, t01, tk1 = big_input(sampling_rate, time_start, time_to_stop)

    print('choose second thing to add')
    signal2, t02, tk2 = big_input(sampling_rate, time_start, time_to_stop)
    signal = sum(signal1, t01, tk1, signal2, t02, tk2, sampling_rate)

    save_to_csv(filename, time_start, time_to_stop, sampling_rate * (time_to_stop - time_start), signal)

elif user_input1 == 3:
    sampling_rate = int(input('enter sampling rate: '))
    time_start = int(input('enter starting time: '))
    time_to_stop = int(input('enter end time: '))

    print('choose what are you going to subtract')
    signal1, t01, tk1 = big_input(sampling_rate, time_start, time_to_stop)

    print('choose second thing to subtract')
    signal2, t02, tk2 = big_input(sampling_rate, time_start, time_to_stop)
    signal = subtract(signal1, t01, tk1, signal2, t02, tk2, sampling_rate)

    save_to_csv(filename, time_start, time_to_stop, sampling_rate * (time_to_stop - time_start), signal)

elif user_input1 == 4:
    sampling_rate = int(input('enter sampling rate: '))
    time_start = int(input('enter starting time: '))
    time_to_stop = int(input('enter end time: '))

    print('choose what are you going to subtract')
    signal1, t01, tk1 = big_input(sampling_rate, time_start, time_to_stop)

    print('choose second thing to subtract')
    signal2, t02, tk2 = big_input(sampling_rate, time_start, time_to_stop)
    signal = multiply(signal1, t01, tk1, signal2, t02, tk2, sampling_rate)
    save_to_csv(filename, time_start, time_to_stop, sampling_rate * (time_to_stop - time_start), signal)

elif user_input1 == 5:
    sampling_rate = int(input('enter sampling rate: '))
    time_start = int(input('enter starting time: '))
    time_to_stop = int(input('enter end time: '))

    print('choose what are you going to subtract')
    signal1, t01, tk1 = big_input(sampling_rate, time_start, time_to_stop)

    print('choose second thing to subtract')
    signal2, t02, tk2 = big_input(sampling_rate, time_start, time_to_stop)
    signal = divide(signal1, t01, tk1, signal2, t02, tk2, sampling_rate)
    save_to_csv(filename, time_start, time_to_stop, sampling_rate * (time_to_stop - time_start), signal)

elif user_input1 == 6:
    time, signal = read_from_csv(filename)
    #draw_graph(signal, )
