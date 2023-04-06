import main

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

constant_sampling()