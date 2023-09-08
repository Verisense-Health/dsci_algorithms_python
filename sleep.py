from download_2025E_data import download_signal, combine_signal
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ppg_val import calc_hr, calc_hr_by_window
from scipy.signal import find_peaks
import neurokit2 as nk
import scipy
from scipy.stats import linregress
from dsci_tools import my_minmax
import pytz
sns.set_style("darkgrid")

BUCKET = "verisense-cd1f868f-eada-44ac-b708-3b83f2aaed73"
USER = "LS2025E"
DEVICE = "210202054E02"
SIGNALS = ["GreenPPG", "RedPPG", "Accel"]
COMBINED_OUT_PATH = f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}"

Path(COMBINED_OUT_PATH).mkdir(parents=True, exist_ok=True)
def load_sleep_labels(file):
    sleep_stages_mat = scipy.io.loadmat(file)
    sleep_stages = sleep_stages_mat['predict_label_transfer']
    sleep_stages = np.array(sleep_stages).flatten()
    return sleep_stages

def main():
    for signal in SIGNALS:
        download_signal(BUCKET, USER, DEVICE, signal, after = "2023-09-01")


    verisense_green_ppg = combine_signal(USER, DEVICE, signal ="GreenPPG", outfile = f"{COMBINED_OUT_PATH}/verisense_green_ppg.csv", use_cache = False, after = "2023-09-01")
    verisense_acc = combine_signal(USER, DEVICE, signal ="Accel", outfile = f"{COMBINED_OUT_PATH}/verisense_acc.csv", use_cache = False, after = "2023-09-01")
    start = 1693956331
    end = 1693986833

    sleep_stages = load_sleep_labels("/Users/lselig/Desktop/verisense/codebase/PhysioNet-Cardiovascular-Signal-Toolbox/lucas_sleep_0906_raw_class1.mat")
    sleep_stages_etime = np.linspace(start, end, sleep_stages.shape[0])
    verisense_green_ppg = verisense_green_ppg[verisense_green_ppg.etime.between(start, end)]
    verisense_acc = verisense_acc[verisense_acc.etime.between(start, end)]
    print(verisense_acc.head())

    tmp_signal, tmp_peaks, tmp_hr, tmp_sqi = calc_hr(verisense_green_ppg,
            25.0,
            do_bandpass=False,
            do_smoothing = False,
            do_median_filter = False,
            ppg_color="green",
            ppg_valname="green",
            ppg_timename="etime",
            device="Verisense"
            )



    hr_by_window = calc_hr_by_window(verisense_green_ppg, tmp_peaks, window = 30, stride = 1)
    hr_by_window = pd.merge(hr_by_window, tmp_sqi, on = ["etime"])
    hr_by_window = hr_by_window[hr_by_window.sqi_range < np.nanmean(hr_by_window.sqi_range) + np.nanstd(hr_by_window.sqi_range)]
    pct_good = np.where(tmp_sqi.sqi_range < np.nanmean(tmp_sqi.sqi_range) + np.nanstd(tmp_sqi.sqi_range))[0].shape[0] / tmp_sqi.shape[0]
    average_hr = np.nanmean(hr_by_window.bpms)
    fig, axs = plt.subplots(4, 1, figsize = (15, 9), sharex = True)
    threshold =np.nanmean(tmp_sqi.sqi_range) + np.nanstd(tmp_sqi.sqi_range)
    for row in tmp_sqi.itertuples():
        if(row.sqi_range > threshold):
            axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "red", lw = 0)
            axs[1].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "red", lw = 0)
        else:
            axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "green", lw = 0)
            axs[1].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "green", lw = 0)

    axs[0].plot(hr_by_window.etime, hr_by_window.bpms, color = "black")
    axs[0].set_ylabel("HR (BPM)")
    axs[1].plot(verisense_green_ppg.etime, verisense_green_ppg.green, color = "black")
    axs[1].set_ylabel("Green PPG")

    axs[2].plot(sleep_stages_etime, sleep_stages, color = "black", drawstyle = "steps-post", marker = ".")
    axs[2].set_ylabel("Sleep Stage")
    axs[3].plot(verisense_acc.etime, verisense_acc.mag, color = "black")
    axs[3].set_ylabel("Accel (g)")
    # map = {1: "Wake", 2: "REM", 3: "Light", 4: "Deep"}
    # axs[2].set_yticklabels([map[x] for x in axs[2].get_yticks()])
    fig.suptitle(f"Lucas sleep 09/06\n"
                 f"Percent clean PPG: {pct_good:.2f}\n"
                 f"Average HR: {average_hr:.2f}")

    plt.show()

if __name__ == "__main__":
    main()
