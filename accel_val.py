from download_2025E_data import download_signal, combine_signal
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

BUCKET = "verisense-cd1f868f-eada-44ac-b708-3b83f2aaed73"
USER = "LS2025E"
DEVICE = "210202054E02"
SIGNALS = ["Accel"]
COMBINED_OUT_PATH = f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}"
Path(COMBINED_OUT_PATH).mkdir(parents=True, exist_ok=True)

def baseline_noise_compare(user, device, accel, axivity_acc, start, end, signal):
    accel = accel[accel.etime.between(start, end)]
    axivity_acc = axivity_acc[axivity_acc.etime.between(start, end)]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"{signal} Comparison\n"
                 f"User: {user}\n"
                 f"Device: {device}\n"
                 f"Duration: {(end - start) / 3600: .2f} hours")
    axs[0].plot(accel.etime, accel[signal].values, label="Verisense", alpha=0.6, color="black")
    axs[0].plot(axivity_acc.etime, axivity_acc[signal].values, label="Axivity", alpha=0.6, color="red")
    axs[0].set_ylabel(f"{signal} (g)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylim([-1.5, 1.5])
    axs[1].set_xlabel(f"{signal} (g)")
    # axs[2].set_xlabel(f"{signal} (g)")
    axs[0].legend()

    # axs[1].hist(accel[signal].values - 1, bins=20, alpha=0.6, color="black", label="Verisense", density=True)
    # axs[1].hist(axivity_acc[signal].values - 1, bins=20, alpha=0.6, color="red", label="Axivity", density=True)

    sns.kdeplot(accel[signal].values, color="black", ax=axs[1], label="Verisense", fill=True)
    sns.kdeplot(axivity_acc[signal].values, color="red", ax=axs[1], label="Axivity", fill=True)
    plt.show()

def parse_axivity(infile):
    df = pd.read_csv(infile, skiprows=4)
    df.columns = ["etime", "x", "y", "z"]
    df["mag"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    # df["etime"] = df["etime"] / 1000
    df = df.sort_values(by="etime")
    return df
def compare_mag(user, device, accel, axivity_acc, start, end, do_hist = False, do_xyz = False, do_resolution = True, verisense_offset = 0):
    accel = accel[accel.etime.between(start, end)]
    accel["etime"] = accel["etime"] + verisense_offset
    axivity_acc = axivity_acc[axivity_acc.etime.between(start, end)]
    if(do_hist):
        fig, axs = plt.subplots(1, 3, figsize = (10, 10))
        fig.suptitle(f"ENMO Comparison\n"
                  f"User: {user}\n"
                  f"Device: {device}")
        axs[0].plot(accel.etime, accel.mag.values - 1, label = "Verisense", alpha = 0.6, color = "black")
        axs[0].plot(axivity_acc.etime, axivity_acc.mag.values - 1, label = "Axivity", alpha = 0.6, color = "red")
        axs[0].set_ylabel("ENMO (g)")
        axs[0].set_xlabel("Time (s)")
        axs[1].set_xlabel("ENMO (g)")
        axs[2].set_xlabel("ENMO (g)")
        axs[0].legend()

        axs[1].hist(accel.mag.values - 1, bins = 100, alpha = 0.6, color = "black", label = "Verisense", density = True)
        axs[1].hist(axivity_acc.mag.values - 1, bins = 100, alpha = 0.6, color = "red", label = "Axivity", density = True)

        sns.kdeplot(accel.mag.values - 1, color = "black", ax = axs[2], label = "Verisense", fill = True, clip = [-1, 4])
        sns.kdeplot(axivity_acc.mag.values - 1, color = "red", ax = axs[2], label = "Axivity", fill = True, clip = [-1, 4])
        plt.show()

    if(do_xyz):
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex = True )
        fig.suptitle(f"ENMO Comparison\n"
                     f"User: {user}\n"
                     f"Device: {device}")
        axs[0].plot(pd.to_datetime(accel.etime, unit = "s"), accel.mag.values - 1, label="Verisense", alpha=0.6, color="black")
        axs[0].plot(pd.to_datetime(axivity_acc.etime, unit = "s"), axivity_acc.mag.values - 1, label="Axivity", alpha=0.6, color="red")
        axs[0].set_ylabel("ENMO (g)")
        axs[0].legend()

        axs[1].plot(pd.to_datetime(accel.etime, unit = "s"), accel.x.values, label="Verisense", alpha=0.6, color="black")
        axs[1].plot(pd.to_datetime(axivity_acc.etime, unit = "s"), axivity_acc.x.values, label="Axivity", alpha=0.6, color="red")
        axs[1].set_ylabel("Acc X (g)")

        axs[2].plot(pd.to_datetime(accel.etime, unit = "s"), accel.y.values, label="Verisense", alpha=0.6, color="black")
        axs[2].plot(pd.to_datetime(axivity_acc.etime, unit = "s"), axivity_acc.y.values, label="Axivity", alpha=0.6, color="red")
        axs[2].set_ylabel("Acc Y (g)")

        axs[3].plot(pd.to_datetime(accel.etime, unit = "s"), accel.z.values, label="Verisense", alpha=0.6, color="black")
        axs[3].plot(pd.to_datetime(axivity_acc.etime, unit = "s"), axivity_acc.z.values, label="Axivity", alpha=0.6, color="red")
        axs[3].set_ylabel("Acc z (g)")
        axs[3].set_xlabel("Time (s)")

        plt.show()

    if(do_resolution):
        res_x = np.digitize(accel.x, bins = np.arange(-8.0, 8.0, 0.001))
        res_y = np.digitize(accel.y, bins = np.arange(-8.0, 8.0, 0.001))
        res_z = np.digitize(accel.z, bins = np.arange(-8.0, 8.0, 0.001))
        a = 1


def main():
    for signal in SIGNALS:
        download_signal(BUCKET, USER, DEVICE, signal)

    verisense_acc = combine_signal(USER, DEVICE, signal ="Accel", outfile = f"{COMBINED_OUT_PATH}/verisense_acc.csv", use_cache = True)
    axivity_acc = parse_axivity("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/acc_range_test/0901_day_axivity.csv")
    start = 1693544487
    compare_mag(USER,
                DEVICE,
                verisense_acc,
                axivity_acc,
                start = start,
                end = np.inf,
                do_hist = False,
                do_xyz = True,
                do_resolution = False,
                verisense_offset = -4)

if __name__ == "__main__":
    main()
