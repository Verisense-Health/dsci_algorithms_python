from download_2025E_data import download_signal, combine_signal
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import neurokit2 as nk
import scipy
sns.set_style("darkgrid")

BUCKET = "verisense-cd1f868f-eada-44ac-b708-3b83f2aaed73"
USER = "LS2025E"
DEVICE = "210202054E02"
SIGNALS = ["GreenPPG", "RedPPG"]
COMBINED_OUT_PATH = f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}"

Path(COMBINED_OUT_PATH).mkdir(parents=True, exist_ok=True)

def parse_polar(polar_path):
    df = pd.read_csv(polar_path)
    date = df.iloc[0]["Date"]
    time = df.iloc[0]["Start time"]
    from datetime import datetime

    mystr = date + " " + time
    start = int(datetime.strptime(mystr, "%d-%m-%Y %H:%M:%S").timestamp()) - 3600 * 5
    df = df[2:]
    df["etime"] = range(start, start + len(df))
    df["hr"] = [float(x) for x in df["Date"]]
    df = df[["etime", "hr"]]
    return df

def parse_shimmer_ppg(shimmer_path):
    df = pd.read_csv(shimmer_path, skiprows = 2)
    df["etime"] = (df["ms"] / 1000) - 3600 * 5
    df["x"] = df["m/(s^2)"] / 9.81
    df["y"] = df["m/(s^2).1"] / 9.81
    df["z"] = df["m/(s^2).2"] / 9.81
    df["mag"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    df["green"] = df["mV"]
    df = df[["etime", "green", "mag"]]
    return df
def parse_verisense_direct_hr(f, signal):
    df = pd.read_csv(f)
    return df
def calc_hr(df, fs, do_bandpass, do_smoothing, ppg_color, ppg_valname, ppg_timename, device):
    ppg_signal = df[[ppg_valname]].values
    processed_signal = nk.ppg_process(ppg_signal, sampling_rate=fs)  # Replace with your actual sampling rate
    # peaks = processed_signal[1]["PPG_Peaks"]
    ppg_signal_clean = processed_signal[0]["PPG_Clean"].values
    if(do_bandpass):
        f1, f2 = 0.5, 8.0  # Bandpass frequency range (Hz)
        # Design the bandpass filter
        order = 3  # Filter order
        nyquist_freq = 0.5 * fs
        low_cutoff = f1 / nyquist_freq
        high_cutoff = f2 / nyquist_freq
        b, a = scipy.signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass')
        # Apply the bandpass filter to the PPG signal
        ppg_signal_bp = scipy.signal.lfilter(b, a, ppg_signal)
    if(do_smoothing):
        pass

    # ppg_signal_bp = np.array([x[0] for x in ppg_signal_bp])
    # peaks, _ = find_peaks(ppg_signal_bp, distance = fs /6)
    # peaks, _ = find_peaks(ppg_signal_bp)
    # blah = hp.process(my_minmax(ppg_signal_bp), sample_rate = 25.0, calc_freq = True)
    ppg_signal_bp = ppg_signal_clean
    # peaks, _ = find_peaks(ppg_signal_bp, height = 0)
    peaks, _ = find_peaks(ppg_signal_bp, height = 0, distance = 25)
    plt.plot(ppg_signal_bp, color=ppg_color, label=f"{ppg_color} ppg")
    plt.scatter(peaks, ppg_signal_bp[peaks], marker="x", color="black", label="peak")
    plt.ylabel("Preprocessed PPG")
    plt.xlabel("Sample")
    plt.title(f"npeaks = {len(peaks)}\n"
              f"duration = {(np.round(df[ppg_timename].max() - df[ppg_timename].min()), 2)[0]:.2f} seconds\n"
              f"BPM = {len(peaks) / (((df[ppg_timename].max() - df[ppg_timename].min())) / 60):.2f}")
    plt.show()
    return ppg_signal_bp, peaks
def compare_ppg(polar_path, shimmer_path, verisense_df, verisense_acc_df, title, shimmer_offset, verisense_ylim, shimmer_ylim, xlim, do_buffer):
    polar_df = parse_polar(polar_path)
    shimmer_df = parse_shimmer_ppg(shimmer_path)
    if(do_buffer):
        start = polar_df.iloc[0].etime - 60*10
        end = polar_df.iloc[-1].etime + 60*10
    else:
        start = polar_df.iloc[0].etime
        end = polar_df.iloc[-1].etime

    shimmer_df["etime"] = shimmer_df["etime"] + shimmer_offset
    verisense_df = verisense_df[(verisense_df["etime"] > start) & (verisense_df["etime"] < end)]
    # verisense_df["etime"] = np.linspace(verisense_df.iloc[0].etime, verisense_df.iloc[-1].etime, num = verisense_df.shape[0])

    verisense_acc_df = verisense_acc_df[(verisense_acc_df["etime"] > start) & (verisense_acc_df["etime"] < end)]


    if(xlim is not None):
        xlim[0] = xlim[0] + 3600 * -5
        xlim[1] = xlim[1] + 3600 * -5
        shimmer_df = shimmer_df[(shimmer_df["etime"] > xlim[0]) & (shimmer_df["etime"] < xlim[1])]
        verisense_df = verisense_df[(verisense_df["etime"] > xlim[0]) & (verisense_df["etime"] < xlim[1])]
        verisense_acc_df = verisense_acc_df[(verisense_acc_df["etime"] > xlim[0]) & (verisense_acc_df["etime"] < xlim[1])]
        polar_df = polar_df[(polar_df["etime"] > xlim[0]) & (polar_df["etime"] < xlim[1])]


    # plt.plot(np.diff(verisense_df.etime.values))
    # plt.show()


    v_signal, v_peaks = calc_hr(verisense_df,
                                fs=25.0,
                                do_bandpass=True,
                                do_smoothing=False,
                                ppg_color="green",
                                ppg_valname="green",
                                ppg_timename="etime",
                                device="Verisense")

    s_signal, s_peaks = calc_hr(shimmer_df,
                                fs=100.0,
                                do_bandpass=True,
                                do_smoothing=False,
                                ppg_color="green",
                                ppg_valname="green",
                                ppg_timename="etime",
                                device="Shimmer")

    window = 30
    stride = 1
    bpms, anchors = [], []
    start = verisense_df.iloc[0].etime + window
    polar_end = polar_df.iloc[-1].etime
    while(start < verisense_df.iloc[-1].etime):
        end = start + window
        peak_timings = verisense_df.etime.values[v_peaks]
        npeaks = np.where((peak_timings > start) & (peak_timings <= end))
        bpm = len(npeaks[0]) / (window / 60)
        anchor = end
        bpms.append(bpm)
        anchors.append(anchor)
        start += stride
        a = 1

    anchors = np.array(anchors)
    verisense_hr = pd.DataFrame({"etime": anchors, "bpms": bpms})
    verisense_hr = verisense_hr[verisense_hr.etime <= polar_end]
    fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)
    axs[0].plot(verisense_hr.etime, verisense_hr.bpms, color = 'C0', label = 'Verisense')

    bpms, anchors = [], []
    start = shimmer_df.iloc[0].etime + window
    while (start < shimmer_df.iloc[-1].etime):
        end = start + window
        peak_timings = shimmer_df.etime.values[s_peaks]
        npeaks = np.where((peak_timings > start) & (peak_timings <= end))
        bpm = len(npeaks[0]) / (window / 60)
        anchor = end

        bpms.append(bpm)
        anchors.append(anchor)
        start += stride
        a = 1


    shimmer_hr = pd.DataFrame({"etime": anchors, "bpms": bpms})
    shimmer_hr = shimmer_hr[shimmer_hr.etime <= polar_end]
    axs[0].plot(shimmer_hr.etime, shimmer_hr.bpms, color = 'C1', label = 'Shimmer')

    start = anchors[0]
    polar_df = polar_df[polar_df.etime.between(start, end)]
    axs[0].plot(polar_df.etime, polar_df.hr, color = 'red', label = 'Polar')
    axs[0].legend()
    fig.suptitle("HR Compare")
    axs[0].set_ylabel("HR (BPM)")
    print(len(polar_df.etime), shimmer_hr.shape[0], verisense_hr.shape[0])

    print("polar shimmer", np.corrcoef(np.array(polar_df.hr), np.array(shimmer_hr.bpms))[0, 1])
    print("polar verisense", np.corrcoef(np.array(polar_df.hr), np.array(verisense_hr.bpms))[0, 1])

    print("polar shimmer mae", np.absolute(np.subtract(polar_df.hr.values, shimmer_hr.bpms)).median())
    print("polar verisense mae", np.absolute(np.subtract(polar_df.hr.values, verisense_hr.bpms)).median())


    axs[1].plot(verisense_df.etime, verisense_df.green, color = 'C0', label = 'Verisense')
    axs[2].plot(shimmer_df.etime, shimmer_df.green, color = 'C1', label = 'Shimmer')
    axs[1].set_ylabel("Verisense Green PPG")
    axs[2].set_ylabel("Shimmer Green PPG")
    plt.show()


    plt.plot(np.diff(verisense_df.etime.values))
    plt.xlabel("Samp")
    plt.ylabel("Time Diff (s)")
    plt.show()

    # wd, m = hp.process_segmentwise(verisense_df.green.values, sample_rate = 25.0, segment_width = 120, segment_overlap = 0.5)
    # wd2, m2 = hp.process_segmentwise(shimmer_df.green.values, sample_rate=100.0, segment_width = 120, segment_overlap = 0.5)

    fig, axs = plt.subplots(4, 1, figsize = (15, 9), sharex = True)
        # for a in axs:
    #     a.set_xlim(pd.to_datetime(xlim[0], unit = "s"), pd.to_datetime(xlim[1], unit = "s"))
    fig.suptitle(title)
    axs[-1].plot(pd.to_datetime(verisense_acc_df.etime, unit = "s"), verisense_acc_df.mag, label = "Verisense")
    axs[-1].plot(pd.to_datetime(shimmer_df.etime, unit = "s"), shimmer_df.mag, label = "Shimmer")
    axs[-1].legend()
    axs[-1].set_ylabel("Magnitude (g)")
    # axs[0].set_ylabel("Verisense Green PPG")
    # axs[0].plot(pd.to_datetime(verisense_df["etime"], unit = "s"), verisense_df["green"], label = "Verisense", color = "C0", alpha = 0.7)
    # axs[0].set_ylim(verisense_ylim)
    # axs[0].plot(np.nan, np.nan, label = "Shimmer", color = "C1", alpha = 0.7)
    # axs[0].legend()
    # tw = axs[0].twinx()
    # tw.grid(False)
    # tw.plot(pd.to_datetime(shimmer_df["etime"], unit = "s"), shimmer_df["green"], label = "Shimmer", color = "C1", alpha = 0.7)
    # tw.set_ylabel("Shimmer Green PPG")
    # tw.set_ylim(shimmer_ylim)

    axs[0].plot(pd.to_datetime(verisense_df["etime"], unit = "s"), verisense_df["green"], label = "Verisense", color = "C0")
    axs[0].set_ylabel("Verisense Green PPG")
    axs[0].set_ylim(verisense_ylim)
    axs[0].set_title(f"Verisense sf {verisense_df.shape[0] / (verisense_df.iloc[-1].etime - verisense_df.iloc[0].etime):.2f} hz")
    axs[1].plot(pd.to_datetime(shimmer_df["etime"], unit = "s"), shimmer_df["green"], label = "Shimmer", color = "C1")
    axs[1].set_ylabel("Shimmer Green PPG")
    axs[1].set_ylim(shimmer_ylim)
    axs[1].set_title(f"Verisense sf {shimmer_df.shape[0] / (shimmer_df.iloc[-1].etime - shimmer_df.iloc[0].etime):.2f} hz")

    axs[2].plot(pd.to_datetime(polar_df.etime, unit = "s"), polar_df.hr, label = "Polar", color = "red")
    axs[2].axhline(polar_df.hr.mean(), color = "black", linestyle = "--", label = "Polar Mean")
    axs[2].set_ylabel("Polar HR (BPM)")
    axs[2].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()




    fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)

    # v_signal = verisense_df.green.values
    # s_signal = shimmer_df.green.values

    fig.suptitle("PPG Compare")
    axs[0].plot(verisense_df.etime, v_signal, label = "Verisense Preproc PPG", alpha = 0.8)
    tw = axs[0].twinx()
    tw.plot(shimmer_df.etime, s_signal, label = "Shimmer Preproc PPG", color = "C1", alpha = 0.8)
    tw.plot(np.nan, np.nan, label = "Verisense Preproc PPG")
    tw.legend()
    axs[0].set_ylabel("Verisense Preproc PPG")
    tw.set_ylabel("Shimmer Preproc PPG")
    tw.grid(False)
    axs[1].plot(verisense_df.etime, v_signal)
    axs[1].set_ylabel("Verisense Preproc PPG")
    axs[1].scatter(verisense_df.etime.values[v_peaks], v_signal[v_peaks], marker=".", color="black", label="peak")

    axs[2].plot(shimmer_df.etime, s_signal, color = "C1")
    axs[2].set_ylabel("Shimmer Preproc PPG")
    axs[2].scatter(shimmer_df.etime.values[s_peaks], s_signal[s_peaks], marker=".", color="black", label="peak")
    plt.legend()
    plt.show()
def compare_hrs(polar_df, verisense_ppg_df, laps, labels, ppg_channel):

    # plt.plot(polar_df.etime, polar_df.hr, label = "Polar")
    start = polar_df.iloc[0].etime + 5*60
    end = polar_df.iloc[-1].etime - 1*60
    print(start, end)
    verisense_ppg_df = verisense_ppg_df[verisense_ppg_df.etime.between(start, end)]



    # plt.plot(pd.to_datetime(polar_df.etime, unit = "s"), polar_df.hr, label = "Polar")
    plt.plot(pd.to_datetime(verisense_ppg_df.etime, unit = "s"), verisense_ppg_df[ppg_channel], label = "Verisense PPG")
    plt.show()

    if(ppg_channel == "green"):
        fs = 25.0

    if(ppg_channel == "red"):
        fs = 100.0

    v_signal, v_peaks = calc_hr(verisense_ppg_df,
                                fs=fs,
                                do_bandpass=True,
                                do_smoothing=False,
                                ppg_color=ppg_channel,
                                ppg_valname=ppg_channel,
                                ppg_timename="etime",
                                device="Verisense")
    window = 30
    stride = 1
    bpms, anchors = [], []
    start = verisense_ppg_df.iloc[0].etime + window
    while (start < verisense_ppg_df.iloc[-1].etime):
        end = start + window
        peak_timings = verisense_ppg_df.etime.values[v_peaks]
        npeaks = np.where((peak_timings > start) & (peak_timings <= end))
        bpm = len(npeaks[0]) / (window / 60)
        anchor = end
        bpms.append(bpm)
        anchors.append(anchor)
        start += stride

    anchors = np.array(anchors)
    verisense_hr = pd.DataFrame({"etime": anchors, "bpms": bpms})
    verisense_hr = verisense_hr[verisense_hr.etime <= verisense_ppg_df.iloc[-1].etime]
    polar_df = polar_df[polar_df.etime.between(verisense_hr.iloc[0].etime, verisense_ppg_df.iloc[-1].etime)]
    naxes = 2
    fig, axs = plt.subplots(naxes, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(verisense_hr.etime, verisense_hr.bpms, color='C0', label='Verisense')
    axs[0].plot(polar_df.etime, polar_df.hr, color='red', label='Polar')
    axs[0].legend()
    fig.suptitle(f"HR Compare - {ppg_channel}")
    axs[0].set_ylabel("HR (BPM)")
    axs[1].plot(verisense_ppg_df.etime, verisense_ppg_df[ppg_channel], color = 'C0', label = 'Verisense')
    axs[1].set_ylabel(f"Verisense {ppg_channel} PPG")

    if(laps is not None and labels is not None):
        for i, lap in enumerate(laps):
            if("Walk" in labels[i]):
                color = "green"
            else:
                color = "purple"
            for j in range(naxes):
                axs[j].axvspan(laps[i][0], laps[i][1], facecolor= color, alpha=0.3, zorder=3, label=labels[i])

        handles, labels = axs[0].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(),
                      loc='upper center', bbox_to_anchor=(0.5, 1.50),
                      ncol=3, fancybox=True, shadow=True)
    print("HR array lens", len(polar_df), len(verisense_hr))
    polar_dropout_idx = np.where(polar_df.hr == 0)[0]
    polar_hr_array = polar_df.hr.values[~polar_dropout_idx]
    verisense_hr_array = verisense_hr.bpms.values[~polar_dropout_idx]
    print("polar verisense correlation", np.corrcoef(polar_hr_array, verisense_hr_array)[0, 1])
    print("polar verisense mae", np.median(np.absolute(np.subtract(polar_hr_array, verisense_hr_array))))
    # print("polar verisense correlation", np.corrcoef(polar_hr_array, verisense_hr_array)[0, 1])
    plt.tight_layout()
    plt.show()

def main():
    for signal in SIGNALS:
        download_signal(BUCKET, USER, DEVICE, signal)

    # verisense_acc = combine_signal(USER, DEVICE, signal ="Accel", outfile = f"{COMBINED_OUT_PATH}/verisense_acc.csv", use_cache = False)
    # polar_df = parse_polar("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/green_ppg_test_walk_run/POLAR_Lucas_Selig_2023-09-01_09-19-52_green_walk_run.csv")
    polar_df = parse_polar("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/red_ppg_test_30min/POLAR_Lucas_Selig_2023-09-01_12-03-31_red_ppg_test.csv")
    # verisense_green_ppg = combine_signal(USER, DEVICE, signal ="GreenPPG", outfile = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/green_ppg_test_walk_run/lselig_verisense_green_ppg_walk_run.csv", use_cache = False)
    verisense_red_ppg = combine_signal(USER, DEVICE, signal="RedPPG", outfile=f"{COMBINED_OUT_PATH}/verisense_red_ppg.csv", use_cache=False)

    walk1 = [1693560240, 1693560720]
    jog = [1693560721, 1693560870]
    walk2 = [1693560900, 1693561320]

    laps = [walk1, jog, walk2]
    labels = ["Walk 1", "Jog", "Walk 2"]

    # compare_hrs(polar_df, verisense_green_ppg, laps, labels)
    compare_hrs(polar_df, verisense_red_ppg, None, None, "red")

    green_ppg = combine_signal(USER, DEVICE, signal="GreenPPG", outfile=f"{COMBINED_OUT_PATH}/verisense_green_ppg.csv", use_cache=False)
    red_ppg = combine_signal(USER, DEVICE, signal="RedPPG", outfile=f"{COMBINED_OUT_PATH}/verisense_red_ppg.csv", use_cache=False)
    accel = combine_signal(USER, DEVICE, signal="Accel", outfile=f"{COMBINED_OUT_PATH}/verisense_acc.csv", use_cache=False)

    compare_ppg(polar_path="/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/green_ppg_test_20min/POLAR_lselig_green_ppg_20min.CSV",
                shimmer_path="/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/green_ppg_test_20min/greenppgtest_Session1_Shimmer_F67E_Calibrated_SD.csv",
                verisense_df=green_ppg,
                verisense_acc_df=accel,
                title="Green PPG Comparison",
                shimmer_offset=2.0,
                verisense_ylim=(7500, 12500),
                shimmer_ylim=(1000, 2000),
                xlim=[1692982269, 1692983366],
                do_buffer=True)


if __name__ == "__main__":
    main()
