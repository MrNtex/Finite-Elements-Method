import os
import matplotlib.pyplot as plt


def plot_max_temperature(path: str, name: str, times: list, max_temps: list) -> None:
    plt.figure()
    plt.plot(times, max_temps, label="Max Temp")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [C]")
    plt.title(f"CPU Heating: {name}")
    plt.grid(True)
    plt.legend()
    plt.savefig(path + "_graph.png")
    plt.close()
