import time
import statistics

def measure_average_time(func, args, repetitions=5):
    times = []

    for _ in range(repetitions):
        start = time.time()
        result = func(*args)
        end = time.time()
        elapsed_time = end - start
        times.append(elapsed_time)

    mean_time = statistics.mean(times)
    standard_deviation = statistics.stdev(times)

    return mean_time, standard_deviation, result