import os
import matplotlib.pyplot as plt

def plot_max_temperature(config_file, times, max_temps):
    plt.figure()
    plt.plot(times, max_temps, label="Max Temp")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [C]")
    plt.title(f"CPU Heating: {os.path.basename(config_file)}")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.splitext(config_file)[0] + "_graph.png")
    plt.close()