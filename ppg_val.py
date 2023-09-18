from download_2025E_data import download_signal, combine_signal, parse_green_ppg, parse_red_ppg
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import neurokit2 as nk
import scipy
from scipy.stats import linregress
from dsci_tools import my_minmax
from parse_external import parse_polar
sns.set_style("darkgrid")

BUCKET = "verisense-cd1f868f-eada-44ac-b708-3b83f2aaed73"
USER = "LS2025E"
DEVICE = "210202054E02"
SIGNALS = ["GreenPPG", "RedPPG"]
COMBINED_OUT_PATH = f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}"

Path(COMBINED_OUT_PATH).mkdir(parents=True, exist_ok=True)


def df_to_mat(verisense_green_ppg, outname):
    ppg = verisense_green_ppg.green.values
    ppg = my_minmax(ppg)
    etime = np.linspace(verisense_green_ppg.iloc[0].etime, verisense_green_ppg.iloc[-1].etime, num = len(ppg))
    # processed_signal, info = nk.ppg_process(ppg, sampling_rate=25.0)  # Replace with your actual sampling rate
    # ppg = processed_signal["PPG_Clean"].values
    print("shape", ppg.shape)
    print("writing")
    df = pd.DataFrame({"val": ppg})
    data_dict = df.to_dict(orient='list')
    # Save the data as a .mat file (adjust the filename)
    outfile = f'/Users/lselig/Desktop/verisense/codebase/PhysioNet-Cardiovascular-Signal-Toolbox/Tools/Sleep_PPG_transfer_learning/{outname}.mat'
    scipy.io.savemat(outfile, data_dict)
    print(f"Wrote file to {outfile} ")
    plt.plot(pd.to_datetime(etime, unit = 's'), df.val)
    plt.title("Lucas Sleep 09/06\n"
              "Duration: {:.2f} hours".format((etime[-1] - etime[0]) / 3600))
    plt.ylabel("Green PPG")
    plt.show()
    return

def calc_sqi(ppg_df, color, window, stride):
    sqi_std = []
    sqi_slope = []
    data_range = []
    anchors = []
    start = ppg_df.iloc[0].etime
    last = ppg_df.iloc[-1].etime
    ppg = ppg_df[color].values

    while(start < last):
        end = start + window
        if(end >= last):
            break
        slice = ppg_df[ppg_df.etime.between(start, end)]
        if(slice.shape[0] == 0):
            sqi_std.append(np.nan)
            sqi_slope.append(np.nan)
            data_range.append(np.nan)
        else:
            sqi_std.append(np.nanstd(slice[color].values))
            sqi_slope.append(linregress(slice.etime.values, slice[color].values).slope)
            data_range.append(np.nanmax(slice[color].values) - np.nanmin(slice[color].values))
        start += stride
        anchors.append(end)
    return pd.DataFrame({"etime": anchors, "sqi_std": sqi_std, "sqi_slope": sqi_slope, "sqi_range": data_range})

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
def calc_hr(df, fs, do_bandpass, do_smoothing, do_median_filter, ppg_color, ppg_valname, ppg_timename, device):
    ppg_signal = df[[ppg_valname]].values.flatten()
    print("Calculating SQI")
    sqi = calc_sqi(df, ppg_color, window = 30, stride = 1)
    fig, axs = plt.subplots(4, 1, figsize = (15, 9), sharex = True)
    axs[0].plot(df.etime.values, ppg_signal, label = "raw", color = "black")
    axs[0].set_ylabel(f"PPG {ppg_valname}")

    axs[1].plot(sqi.etime, sqi.sqi_std, color = "C0", label = "sqi")
    axs[1].set_ylabel("STD")

    axs[2].plot(sqi.etime, sqi.sqi_slope, color = "C0", label = "sqi")
    axs[2].set_ylabel("SLOPE")

    threshold =np.nanmean(sqi.sqi_range) + np.nanstd(sqi.sqi_range)
    for row in sqi.itertuples():
        if(row.sqi_range > threshold):
            axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "red", lw = 0)
        else:
            axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "green", lw = 0)
    # for j in range(0, len(sqi), 30):
    #     row = sqi.iloc[j]
    #     if(row.sqi_range > threshold):
            # axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "red", lw = 0)
            # axs[3].axvspan(row.etime -  15, row.etime + 15, alpha = 0.5, color = "red", lw = 0)
        # else:
            # axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "green", lw = 0)
            # axs[3].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "green", lw = 0)

    axs[3].plot(sqi.etime, sqi.sqi_range)
    axs[3].axhline(np.nanmean(sqi.sqi_range), ls = "--", color = "gray", label = "mean")
    axs[3].axhline(np.nanmean(sqi.sqi_range) + np.nanstd(sqi.sqi_range), label = "mean + std", color = "red", ls = "--")
    axs[3].legend()
    axs[3].set_ylabel("MAX - MIN")
    # axs[0].plot(sqi.etime, result, drawstyle = "steps-post", color = "C3", label = "sqi")
    # axs[3].plot(sqi.etime, result, drawstyle = "steps-post", color = "C3", label = "sqi")


    if(do_median_filter):
        ppg_signal = scipy.signal.medfilt(ppg_signal, kernel_size=25)
        axs[0].plot(ppg_signal, color = "red", label = "smoothed")
    # if(do_smoothing):
    #     Apply the moving average filter
    #     window_size = 4
    #     ppg_signal = np.convolve(ppg_signal, np.ones(window_size)/window_size, mode='same')
    # plt.ylabel(f"PPG {ppg_valname}")
    plt.legend()
    plt.show()

    if(do_smoothing):
        from scipy.signal import savgol_filter
        ppg_signal_savgol = savgol_filter(ppg_signal, 51, 3)  # window size 51, polynomial order 3
        plt.plot(ppg_signal_savgol, color = "purple", label = "smoothed", alpha = 0.5)
        plt.plot(ppg_signal, color = "black", label = "orig", alpha = 0.5)
        plt.legend()
        plt.ylabel("Red PPG")
        plt.xlabel("Sample")
        plt.show()
        # pass
        ppg_signal = ppg_signal_savgol
    processed_signal, info = nk.ppg_process(ppg_signal, sampling_rate=fs)  # Replace with your actual sampling rate
    ppg_hr = processed_signal["PPG_Rate"]
    is_peak = processed_signal["PPG_Peaks"].values
    ppg_raw = processed_signal["PPG_Raw"].values
    ppg_clean = processed_signal["PPG_Clean"].values
    # peaks = processed_signal[1]["PPG_Peaks"]
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



    # ppg_signal_bp = np.array([x[0] for x in ppg_signal_bp])
    # peaks, _ = find_peaks(ppg_signal_bp, distance = fs /6)
    # peaks, _ = find_peaks(ppg_signal_bp)
    # blah = hp.process(my_minmax(ppg_signal_bp), sample_rate = 25.0, calc_freq = True)
    # green
    # 200 BPM
    # 3.33 BP second
    # 100 Samples per sec
    # need 100 / 3.33 spacing between
    bpm_max = 200
    bpsecond_max = bpm_max / 60
    distance = fs / bpsecond_max


    if(ppg_color == "red"):
        # peaks_custom, _ = find_peaks(ppg_clean, height = -100, distance = distance)
        peaks_custom, _ = find_peaks(ppg_clean, distance = distance)
        # peaks_custom, _ = find_peaks(ppg_clean, height = 0)
    else:
        peaks_custom, _ = find_peaks(ppg_clean, height=0)
    fig, axs = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axs[0].plot(ppg_raw)
    axs[1].plot(ppg_clean)
    # axs[2].plot(ppg_med_filt)
    peak_idxs = np.where(is_peak > 0)

    axs[0].scatter(peaks_custom, ppg_raw[peaks_custom], marker="x", color="red")
    axs[1].scatter(peaks_custom, ppg_clean[peaks_custom], marker="x", color="red")
    # axs[2].scatter(peaks_custom, ppg_med_filt[peaks_custom], marker="x", color="red")

    axs[0].scatter(peak_idxs, ppg_raw[peak_idxs], marker="x", color="black")
    axs[1].scatter(peak_idxs, ppg_clean[peak_idxs], marker="x", color="black")
    # axs[2].scatter(peak_idxs, ppg_med_filt[peak_idxs], marker="x", color="black")
    # axs[2].scatter(range(len(processed_signal["PPG_Rate"])), processed_signal["PPG_Rate"])
    # axs[3].plot(processed_signal["PPG_Peaks"])

    axs[0].set_ylabel(f"PPG {ppg_valname} Raw")
    axs[1].set_ylabel(f"PPG {ppg_valname} Clean")

    # plt.plot(ppg_clean, color=ppg_color, label=f"{ppg_color} ppg")
    # plt.scatter(peaks, ppg_clean[peaks], marker="x", color="black", label="peak")
    # plt.ylabel("Preprocessed PPG")
    # plt.xlabel("Sample")
    plt.title(f"npeaks = {len(peaks_custom)}\n"
              f"duration = {(np.round(df[ppg_timename].max() - df[ppg_timename].min()), 2)[0]:.2f} seconds\n"
              f"BPM = {len(peaks_custom) / (((df[ppg_timename].max() - df[ppg_timename].min())) / 60):.2f}")
    plt.show()
    return ppg_clean, peaks_custom, ppg_hr, sqi

def calc_hr_by_window(df, peaks, window, stride):
    bpms, anchors = [], []
    start = df.iloc[0].etime + window
    while (start < df.iloc[-1].etime):
        end = start + window
        peak_timings = df.etime.values[peaks]
        npeaks = np.where((peak_timings > start) & (peak_timings <= end))
        bpm = len(npeaks[0]) / (window / 60)
        anchor = end

        bpms.append(bpm)
        anchors.append(anchor)
        start += stride
    ret = pd.DataFrame({"etime": anchors, "bpms": bpms})
    return ret
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


    v_signal, v_peaks, v_hr, v_sqi = calc_hr(verisense_df,
                                fs=25.0,
                                do_bandpass=True,
                                do_smoothing=False,
                                ppg_color="green",
                                ppg_valname="green",
                                ppg_timename="etime",
                                device="Verisense")

    s_signal, s_peaks, s_hr, s_sqi = calc_hr(shimmer_df,
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
def compare_hrs(polar_df, verisense_ppg_df, laps, labels, ppg_channel, trial_name):

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

    v_signal, v_peaks, v_hr, v_sqi = calc_hr(verisense_ppg_df,
                                fs=fs,
                                do_bandpass=True,
                                do_smoothing=True,
                                do_median_filter=False,
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
    axs[0].plot(verisense_hr.etime, verisense_hr.bpms, color='black', label='Verisense')
    # axs[0].plot(np.linspace(polar_df.iloc[0].etime, polar_df.iloc[-1].etime, num = len(v_hr)), v_hr, color = 'C3', label = 'NK Verisense')
    axs[0].plot(polar_df.etime, polar_df.hr, color='purple', label='Polar')
    axs[0].legend()
    fig.suptitle(f"HR Compare - {ppg_channel}")
    sqi = calc_sqi(verisense_ppg_df, ppg_channel, window = 30, stride = 1)

    threshold =np.nanmean(sqi.sqi_range) + np.nanstd(sqi.sqi_range)
    for row in sqi.itertuples():
        if(row.sqi_range > threshold):
            axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "red", lw = 0)
            axs[1].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "red", lw = 0)
        else:
            axs[0].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "green", lw = 0)
            axs[1].axvspan(row.etime - 15, row.etime + 15, alpha = 0.5, color = "green", lw = 0)

    axs[0].set_ylabel("HR (BPM)")
    axs[1].plot(verisense_ppg_df.etime, verisense_ppg_df[ppg_channel], color = 'black', label = 'Verisense')
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
    good_polar_idx = np.where(polar_df.hr != 0)[0][:-1]
    verisense_hr = pd.merge(verisense_hr,sqi, on = ["etime"])
    # polar_hr_array = pd.merge(polar_df, sqi, on = ["etime"])
    if(len(good_polar_idx) > 0):
        # polar_hr_array = polar_df.hr.values[good_polar_idx]
        # verisense_hr_array = verisense_hr.bpms.values[good_polar_idx]
        polar_df = polar_df.iloc[good_polar_idx]
        verisense_hr = verisense_hr.iloc[good_polar_idx]

    # ignore bad sqi windows
    good_sqi = np.where(verisense_hr.sqi_range < threshold)[0][:-1]
    verisense_hr_array = verisense_hr.iloc[good_sqi].bpms.values
    polar_hr_array = polar_df.iloc[good_sqi].hr.values
    # polar_hr_array = polar_hr_array[good_sqi]
    print("polar verisense correlation", np.corrcoef(polar_hr_array, verisense_hr_array)[0, 1])
    print("polar verisense mae", np.mean(np.absolute(np.subtract(polar_hr_array, verisense_hr_array))))
    # print("polar verisense correlation", np.corrcoef(polar_hr_array, verisense_hr_array)[0, 1])
    plt.tight_layout()
    plt.show()

    # bland altman
    plot_bland_altman(polar_hr_array, verisense_hr_array, trial_name)

def plot_bland_altman(polar_hr, verisense_hr, trial_name):
    # x axis is mean of two measurements
    xaxis = (polar_hr + verisense_hr) / 2

    # y axis is polar - verisense
    yaxis = polar_hr - verisense_hr

    dline = np.mean(yaxis)
    dline_plus_std = dline + 1.96 * np.std(yaxis)
    dline_minus_std = dline - 1.96 * np.std(yaxis)

    plt.hlines(dline, xmin = min(xaxis), xmax = max(xaxis), color = "black", linestyle = "--", label = "D")
    plt.hlines(dline_plus_std, xmin = min(xaxis), xmax = max(xaxis), color = "black", linestyle = "--", alpha = 0.5, label = "D + 1.96*std")
    plt.hlines(dline_minus_std, xmin = min(xaxis), xmax = max(xaxis), color = "black", linestyle = "--", alpha = 0.5, label = "D - 1.96*std")
    plt.legend()
    plt.ylabel("Polar HR - Verisense HR")
    plt.xlabel("Mean HR (BPM)")
    plt.title(f"Bland-Altman\n{trial_name}")
    plt.scatter(xaxis, yaxis)
    plt.tight_layout()
    plt.show()

def main():
    # for signal in SIGNALS:
    #     download_signal(BUCKET, USER, DEVICE, signal)

    # verisense_acc = combine_signal(USER, DEVICE, signal ="Accel", outfile = f"{COMBINED_OUT_PATH}/verisense_acc.csv", use_cache = False)
    # polar_df = parse_polar("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/green_ppg_test_20min/POLAR_lselig_green_ppg_20min.CSV")
    polar_df = parse_polar("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/red_ppg_test_30min/POLAR_Lucas_Selig_2023-09-01_12-03-31_red_ppg_test.csv")
    # polar_df = parse_polar("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/green_ppg_test_walk_run/POLAR_Lucas_Selig_2023-09-01_09-19-52_green_walk_run.csv")
    verisense_green_ppg = combine_signal(USER, DEVICE, signal ="GreenPPG", outfile = f"{COMBINED_OUT_PATH}/verisense_green_ppg.csv", use_cache = False, after = "2023-08-01")
    verisense_red_ppg = combine_signal(USER, DEVICE, signal ="RedPPG", outfile = f"{COMBINED_OUT_PATH}/verisense_red_ppg.csv", use_cache = True, after = "2023-08-01")
    FS = 100.0

    start = 1693956331
    end = 1693986833
    start = polar_df.iloc[0].etime
    end = polar_df.iloc[-1].etime
    verisense_green_ppg = verisense_green_ppg[verisense_green_ppg.etime.between(start, end)]
    verisense_red_ppg = verisense_red_ppg[verisense_red_ppg.etime.between(start, end)]

    PPG_SIGNAL  = verisense_red_ppg

    # sqi = calc_sqi(verisense_green_ppg, "green", window = 30, stride = 1)
    tmp_signal, tmp_peaks, tmp_hr, tmp_sqi = calc_hr(PPG_SIGNAL,
            FS,
            do_bandpass=False,
            do_smoothing = True,
            do_median_filter = False,
            ppg_color="red",
            ppg_valname="red",
            ppg_timename="etime",
            device="Verisense"
            )

    hr_by_window = calc_hr_by_window(PPG_SIGNAL, tmp_peaks, window = 30, stride = 1)
    hr_by_window = pd.merge(hr_by_window, tmp_sqi, on = ["etime"])
    hr_by_window = hr_by_window[hr_by_window.sqi_range < np.nanmean(hr_by_window.sqi_range) + np.nanstd(hr_by_window.sqi_range)]
    pct_good = np.where(tmp_sqi.sqi_range < np.nanmean(tmp_sqi.sqi_range) + np.nanstd(tmp_sqi.sqi_range))[0].shape[0] / tmp_sqi.shape[0]
    average_hr = np.nanmean(hr_by_window.bpms)
    compare_hrs(polar_df, verisense_red_ppg, None, None, "red", trial_name = "Red PPG 20 min stationary")
    fig, axs = plt.subplots(2, 1, figsize = (15, 9), sharex = True)
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
    fig.suptitle(f"Lucas sleep 09/06\n"
                 f"Percent clean PPG: {pct_good:.2f}\n"
                 f"Average HR: {average_hr:.2f}")

    plt.show()

    df_to_mat(verisense_green_ppg, outname = "lucas_sleep_0906_raw")

    walk1 = [1693560240, 1693560720]
    jog = [1693560721, 1693560870]
    walk2 = [1693560900, 1693561320]

    laps = [walk1, jog, walk2]
    labels = ["Walk 1", "Jog", "Walk 2"]

    # compare_hrs(polar_df, verisense_green_ppg, laps, labels, "green", trial_name = "Green PPG 20 min stationary")
    compare_hrs(polar_df, verisense_green_ppg, None, None, "green", trial_name = "Green PPG walk/run")

    green_ppg = combine_signal(USER, DEVICE, signal="GreenPPG", outfile=f"{COMBINED_OUT_PATH}/verisense_green_ppg.csv", use_cache=True)
    red_ppg = combine_signal(USER, DEVICE, signal="RedPPG", outfile=f"{COMBINED_OUT_PATH}/verisense_red_ppg.csv", use_cache=True)
    accel = combine_signal(USER, DEVICE, signal="Accel", outfile=f"{COMBINED_OUT_PATH}/verisense_acc.csv", use_cache=True)

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
    # green_ppg = parse_green_ppg("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_green/230911_012101_GreenPPG.csv", show_plot = True)
    # red_ppg = parse_red_ppg("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_red/230911_101009_RedPPG.csv", show_plot = True)

    main()
    green_ppg = parse_green_ppg("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_green/230911_133453_GreenPPG.csv", show_plot = True)
    red_ppg = parse_red_ppg("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_red/230911_150717_RedPPG.csv", show_plot = True)

    plt.plot(green_ppg)
