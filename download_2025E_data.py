import awswrangler as wr
import heartpy as hp
from scipy.signal import find_peaks
import scipy
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
sns.set_style("darkgrid")
import datetime
import dateutil
import boto3
from pathlib import Path
from dateutil import parser
def read_line(infile, linei):
    with open(infile, "r") as f:
        for i, line in enumerate(f):
            if(i == linei - 1):
                return line
    return None

def parse_accel(infile):
    df = pd.read_csv(infile, skiprows=9)
    df.columns = ["etime", "x", "y", "z"]
    df = df.sort_values(by = "etime")
    df["x"] = (df["x"] / 256)
    df["y"] = (df["y"] / 256)
    df["z"] = (df["z"] / 256)
    mag = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    df["etime"] = df["etime"] / 1000
    df["orig_etime"] = df["etime"]
    etime = df.orig_etime.values
    orig_etime = df.orig_etime.values
    copy_df = df.copy()

    unique_ts = np.unique(etime)
    for u in unique_ts:
        # if(len(np.where(etime == u)[0]) <= 1):
        #     a = 1
        start = np.where(etime == u)[0][0]
        end = np.where(etime == u)[0][-1]
        chunk_ts = np.linspace(u, u + 1, end - start, endpoint = False)
        copy_df["etime"][start:end] = chunk_ts
        a = 1

    df = copy_df
    etime = df.etime
    df = df[["x", "y", "z"]]
    #orientation for shimmer imu
    # df = df.dot(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))
    df = df.dot(np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]))
    df = df.rename(columns={0: "x", 1: "y", 2: "z"})
    df["mag"] = mag
    df["etime"] = etime
    df['orig_etime'] = orig_etime
    df = df.sort_values(by="etime")
    df = df.drop_duplicates()
    return df

def parse_green_ppg(infile):
    df = pd.read_csv(infile, skiprows=9)
    df.columns = ["etime", "green"]
    df["etime"] = df["etime"] / 1000
    df = df.sort_values(by="etime")
    return df

def parse_red_ppg(infile):
    df = pd.read_csv(infile, skiprows=9)
    df.columns = ["etime", "red"]
    df["etime"] = df["etime"] / 1000
    df = df.sort_values(by="etime")
    return df
def parse_2025e(infile, signal):
    if(signal == "Accel"):
        df = parse_accel(infile)
    elif(signal == "GreenPPG"):
        df = parse_green_ppg(infile)
    elif(signal == "RedPPG"):
        df = parse_red_ppg(infile)
    # elif(signal == "Temperature"):
    #     df = parse_temperature(infile)
    # elif(signal == "BloodOxygenLevel"):
    #     df = parse_blood_oxygen_level(infile)
    # elif(signal == "Step"):
    #     df = parse_step(infile)
    else:
        raise ValueError("Signal not recognized")
        return -1
    return df


def parse_axivity(infile):
    df = pd.read_csv(infile, skiprows=4)
    df.columns = ["etime", "x", "y", "z"]
    df["mag"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    # df["etime"] = df["etime"] / 1000
    df = df.sort_values(by="etime")
    return df
# signals = ["Accel", "GreenPPG", "Temperature", "BloodOxygenLevel", "Step", "RedPPG"]
signals = ["Accel", "GreenPPG", "RedPPG", "HeartRate"]
# signals = ["Accel"]
user = "LS2025E"
# user = "Joseph2025E"
device = "210202054E02"
# device = "210202054E00"


# Specify your AWS credentials (make sure they're properly configured)
aws_access_key = "AKIAR2C2O5V35DS42JAQ"
aws_secret_key = "pmwJNqKpGHegFR1U2Qr7ZLFJwfjLXiVcOxFCBlfa"

def download_signal(bucket,
                    user,
                    device,
                    signal):

    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    # List objects in the bucket that contain the specified substring
    objects_with_substring = wr.s3.list_objects(path=f"s3://{bucket}/1/{user}/{device}/ParsedFiles/", suffix=f"{signal}.csv", boto3_session=session)
    for obj in objects_with_substring:
        print("Object Key:", obj)
        saveloc = Path(f"data/{user}/{device}/{signal}")
        if(not saveloc.exists()):
            # print("Downloading...")
            Path(saveloc).mkdir(parents=True, exist_ok=True)
        if(not Path(f"{str(saveloc)}/{obj.split('/')[-1]}").exists()):
            wr.s3.download(path=obj, local_file=f"{str(saveloc)}/{obj.split('/')[-1]}", boto3_session=session)
            print(f"Downloaded: {obj}")
        else:
            print(f"Already downloaded: {obj}")

def parse_verisense_direct_hr(f, signal):
    df = pd.read_csv(f)
    a = 1
def combine_signal(user, device, signal, outfile, use_cache):
    if(use_cache):
        return pd.read_csv(outfile)
    files = Path(f"data/{user}/{device}/{signal}").glob("*.csv")
    dfs = []
    for f in files:
        if("HeartRate" not in signal):
            dfs.append(parse_2025e(f, signal))
        else:
            dfs.append(parse_verisense_direct_hr(f, signal))
    df = pd.concat(dfs)
    df = df.sort_values(by = "etime")
    df = df.drop_duplicates()
    df.to_csv(outfile, index = False)
    return df

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


def my_minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
def calc_hr(df, fs, do_bandpass, do_smoothing, ppg_color, ppg_valname, ppg_timename, device):
    ppg_signal = df[[ppg_valname]].values
    # print(ppg_signal.shape)
    #
    processed_signal = nk.ppg_process(ppg_signal, sampling_rate=fs)  # Replace with your actual sampling rate
    #
    # # Get heart rate from processed signal
    peaks = processed_signal[1]["PPG_Peaks"]
    a = 1
    ppg_signal_clean = processed_signal[0]["PPG_Clean"].values
    # plt.plot(ppg_signal_clean, color=ppg_color, label=f"{ppg_color} ppg")
    # plt.scatter(peaks, ppg_signal_clean[peaks], marker="x", color="black", label="peak")
    # plt.ylabel("Preprocessed PPG")
    # plt.xlabel("Sample")
    # npeaks = len(peaks)
    # bpm = len(peaks) / ( (df[ppg_timename].max() - df[ppg_timename].min()) / 60)
    #
    # plt.title(f"{device} -- neurokit2\n"
    #           f"npeaks = {npeaks}\n"
    #           f"duration = {(np.round(df[ppg_timename].max() - df[ppg_timename].min(), 2)):.2f} seconds\n"
    #           f"BPM = {bpm:.2f}")
    # plt.show()
    # return ppg_signal_clean, peaks
    # do_detrend = True
    # if(do_detrend):
    #     ppg_signal = scipy.signal.detrend(ppg_signal)
    if(do_bandpass):
        f1, f2 = 0.5, 8.0  # Bandpass frequency range (Hz)
        # f1, f2 = 0.2, 0.5  # Bandpass frequency range (Hz)
        # Design the bandpass filter
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
    # return len(peaks) / (((df[ppg_timename].max() - df[ppg_timename].min())) / 60)
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



bucket = "verisense-cd1f868f-eada-44ac-b708-3b83f2aaed73"
for signal in signals:
    download_signal(bucket, user, device, signal)

# polar_df = parse_polar("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/green_ppg_test_walk_run/Lucas_Selig_2023-09-01_09-19-52_green_walk_run.csv")
polar_df = parse_polar("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/red_ppg_test_30min/Lucas_Selig_2023-09-01_12-03-31_red_ppg_test.csv")
# verisense_green_ppg = combine_signal(user, device, signal ="GreenPPG", outfile = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/green_ppg_test_walk_run/lselig_verisense_green_ppg_walk_run.csv", use_cache = False)
verisense_red_ppg = combine_signal(user, device, signal ="RedPPG", outfile = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/red_ppg_test_30min/lselig_verisense_red_ppg_30min.csv", use_cache = False)
# verisense_hr_jointcorp = combine_signal(user, device, signal ="HR", outfile = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/green_ppg_test_walk_run/jointcorp_hr_walk_run.csv", use_cache = False)
# plt.plot(pd.to_datetime(verisense_red_ppg.etime, unit = "s"), verisense_red_ppg.red)
# plt.show()

walk1 = [1693560240, 1693560720]
jog = [1693560721, 1693560870]
walk2 = [1693560900, 1693561320]

laps = [walk1, jog, walk2]
labels = ["Walk 1", "Jog", "Walk 2"]
# compare_hrs(polar_df, verisense_green_ppg, laps, labels)
compare_hrs(polar_df, verisense_red_ppg, None, None, "red")
a = 1

green_ppg = combine_signal(user, device, signal ="GreenPPG", outfile = "/Users/lselig/Documents/joint_corp/watch_green_ppg.csv", use_cache = False)
red_ppg = combine_signal(user, device, signal ="RedPPG", outfile = "/Users/lselig/Documents/joint_corp/watch_red_ppg.csv", use_cache = False)
accel = combine_signal(user, device, signal ="Accel", outfile = "/Users/lselig/Documents/joint_corp/watch_accel.csv", use_cache = True)

compare_ppg(polar_path = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/green_ppg_test_20min/lselig_polar_green_ppg_test1.CSV",
            shimmer_path = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/green_ppg_test_20min/greenppgtest_Session1_Shimmer_F67E_Calibrated_SD.csv",
            verisense_df = green_ppg,
            verisense_acc_df = accel,
            title = "Green PPG Comparison",
            shimmer_offset = 2.0,
            verisense_ylim = (7500, 12500),
            shimmer_ylim = (1000, 2000),
            xlim = [1692982269, 1692983366],
            do_buffer = True)
            # xlim = None)


# accel = pd.read_csv("/Users/lselig/Documents/joint_corp/watch_accel.csv")
# accel = accel.drop_duplicates(subset = ["etime"])
# accel.to_csv("/Users/lselig/Documents/joint_corp/watch_accel.csv", index = False)
# green_ppg = pd.read_csv("/Users/lselig/Documents/joint_corp/watch_green_ppg.csv")
# red_ppg = pd.read_csv("/Users/lselig/Documents/joint_corp/watch_red_ppg.csv")


# green_measurements = np.where(np.array(np.diff(green_ppg.etime.values) < 1))[0]
# chunk_ts = []
# chunk_samp = []
# green_times = green_ppg.etime.values
# last_chunk_time = 0
# for i, samp in enumerate(green_times):
#     diff = green_times[i + 1] - green_times[i]
#     print(diff)
#     if(diff > 1 and chunk_ts[-1] - chunk_ts[0] > 40):
#         print(diff)
#         duration = chunk_ts[-1] - chunk_ts[0]
#         plt.title(f"Duration: {duration:.2f} seconds\n"
#                   f"Time since last chunk: {chunk_ts[0] - last_chunk_time:.2f} seconds")
#         plt.plot(chunk_ts, chunk_samp)
#         plt.show()
#         last_chunk_time = chunk_ts[-1]
#         chunk_ts = []
#         chunk_samp = []
#     else:
#         chunk_samp.append(green_ppg.iloc[i].green)
#         chunk_ts.append(green_ppg.iloc[i].etime)

fig, axs = plt.subplots(2, 1, figsize = (15, 8), sharex = True)
fig.suptitle(f"{user} {device}\n"
             f"PPG Green and Red")
axs[0].plot(pd.to_datetime(green_ppg.etime, unit = "s"), green_ppg.green, label = "PPG Green", color = "green")
# axs[1].plot(pd.to_datetime(green_ppg.etime[:-1], unit = "s"), np.diff(green_ppg.etime.values) / 60)
# axs[1].set_ylabel("Time Delta (m)")
#
axs[1].plot(pd.to_datetime(red_ppg.etime, unit = "s"), red_ppg.red, label = "PPG Red", color = "red")
 # axs[-1].plot(pd.to_datetime(accel.etime, unit = "s"), accel.mag, label = "Accel Mag (g)", color = "black")
plt.show()
#11:50 start
# print(accel.head(50))
# start = 1692712800
start = np.inf * -1
axivity_acc = parse_axivity("/Users/lselig/Desktop/91647_0000000000_axivity.csv")
# axivity_acc = parse_axivity("/Users/lselig/Desktop/baseline_axivity.csv")

accel = accel[accel.etime.values > start]
# plt.plot(accel.etime.values, accel.x.values)
# plt.show()
axivity_acc = axivity_acc[axivity_acc.etime.values > start]

# print(accel[["x", "y", "z"]].describe())
# print(axivity_acc[["x", "y", "z"]].describe())

# start = 1692358778
# end = 1692462203
# print((end - start) / 3600)


def compare_mag(user, device, accel, axivity_acc, start, end, do_hist = False, do_xyz = False, do_resolution = True):
    accel = accel[accel.etime.between(start, end)]
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
        # axs[2].hist(accel.mag.values - 1, bins = 100, alpha = 0.6, color = "black", label = "Verisense", density = True)
        # axs[2].hist(axivity_acc.mag.values - 1, bins = 100, alpha = 0.6, color = "red", label = "Axivity", density = True)


        plt.show()
    if(do_xyz):
        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex = True )
        fig.suptitle(f"ENMO Comparison\n"
                     f"User: {user}\n"
                     f"Device: {device}")
        axs[0].plot(accel.etime, accel.mag.values - 1, label="Verisense", alpha=0.6, color="black")
        axs[0].plot(axivity_acc.etime, axivity_acc.mag.values - 1, label="Axivity", alpha=0.6, color="red")
        axs[0].set_ylabel("ENMO (g)")
        axs[0].legend()

        axs[1].plot(accel.etime, accel.x.values, label="Verisense", alpha=0.6, color="black")
        axs[1].plot(axivity_acc.etime, axivity_acc.x.values, label="Axivity", alpha=0.6, color="red")
        axs[1].set_ylabel("Acc X (g)")

        axs[2].plot(accel.etime, accel.y.values, label="Verisense", alpha=0.6, color="black")
        axs[2].plot(axivity_acc.etime, axivity_acc.y.values, label="Axivity", alpha=0.6, color="red")
        axs[2].set_ylabel("Acc Y (g)")

        axs[3].plot(accel.etime, accel.z.values, label="Verisense", alpha=0.6, color="black")
        axs[3].plot(axivity_acc.etime, axivity_acc.z.values, label="Axivity", alpha=0.6, color="red")
        axs[3].set_ylabel("Acc z (g)")
        axs[3].set_xlabel("Time (s)")

        plt.show()

    if(do_resolution):
        res_x = np.digitize(accel.x, bins = np.arange(-8.0, 8.0, 0.001))
        res_y = np.digitize(accel.y, bins = np.arange(-8.0, 8.0, 0.001))
        res_z = np.digitize(accel.z, bins = np.arange(-8.0, 8.0, 0.001))
        a = 1


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

def calc_fs_sliding(window, stride, df, outfile):
    start = df.iloc[0].orig_etime + 2*window
    end = df.iloc[-1].orig_etime - 2*window
    print(start, end)
    nsamples = []
    anchors = []
    for t in np.arange(start, end, stride):
        print(t)
        nsamples.append(df[df.orig_etime.between(t, t + window)].shape[0])
        anchors.append(np.nanmedian(df[df.orig_etime.between(t, t + window)].orig_etime))


    df = pd.DataFrame({"nsamples": nsamples, "anchors": anchors})
    if(outfile is not None):
        df.to_csv(outfile, index = False)
    return df

# accel = accel[accel.etime.between(1692397800, 1692397800 + 3600 * 6)]
# axivity_acc = axivity_acc[axivity_acc.etime.between(1692397800, 1692397800 + 3600 * 6)]

# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692714232, end = 1692718709, signal = "x")
# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692714232, end = 1692718709, signal = "y")
# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692714232, end = 1692718709, signal = "z")
# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692714232, end = 1692718709, signal = "mag")

# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692384065, end = 1692393620, signal = "x")
# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692384065, end = 1692393620, signal = "y")
# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692384065, end = 1692393620, signal = "z")
# baseline_noise_compare(user, device, accel, axivity_acc, start = 1692384065, end = 1692393620, signal = "mag")
compare_mag(user, device, accel, axivity_acc, start, np.inf, do_hist = False, do_xyz = True, do_resolution = True)


fig, axs = plt.subplots(4, 1, figsize = (10, 10), sharex = True)
print(accel[["x", "y", "z"]].describe())
# for e in accel.etime:
#     print(e, pd.to_datetime(e, unit = "s"))
# axs[0].plot(pd.to_datetime(accel.etime, unit = "s"), accel.mag, label = "watch mag", alpha = 0.6)
fig.suptitle(f"{user} {device}")
axs[0].plot(pd.to_datetime(accel.etime, unit = 's'), accel.x, label = "watch x", alpha = 0.6)
axs[0].plot(pd.to_datetime(accel.etime, unit = 's'), accel.y, label = "watch y", alpha = 0.6)
axs[0].plot(pd.to_datetime(accel.etime, unit = 's'), accel.z, label = "watch z", alpha = 0.6)
axs[0].legend()
axs[0].set_ylabel("Acceleration (g)")
axs[1].plot(pd.to_datetime(accel.etime, unit = "s"), accel.mag.values - 1,  label = "watch mag", color = "black")
axs[1].set_ylabel("ENMO (g)")
axs[2].plot(pd.to_datetime(accel.etime.values[:-1], unit = "s"), np.diff(accel.etime), label = "Interp", alpha = 0.6)
axs[2].set_ylabel("Timestamp Difference (s)")
window, stride = 300, 30
fs_info = pd.read_csv("/Users/lselig/Desktop/fs_info.csv")
axs[3].set_title(f"Window = {window}, Stride = {stride}")
axs[3].plot(pd.to_datetime(fs_info.anchors, unit = "s"), fs_info.nsamples.values / window, label = "nsamples", color = "red")
axs[3].set_title(f"Sampling Rate across {window} second windows(Hz)\n"
                 f"Average: {np.nanmean(fs_info.nsamples.values / window):.2f}\n"
                 f"Min: {np.min(fs_info.nsamples.values / window):.2f}\n"
                 f"Max: {np.max(fs_info.nsamples.values / window):.2f}\n")
axs[3].set_ylabel("Sampling Rate (Hz)")

# plt.plot(pd.to_datetime(axivity_acc.etime, unit = "s"), axivity_acc.mag, label = "axivity mag", alpha = 0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"/Users/lselig/Desktop/accel_{user}_{device}.png")
plt.show()
# green_ppg = combine_signal(user, device, signal ="GreenPPG")
# red_ppg = combine_signal(user, device, signal ="RedPPG")
# fig, axs = plt.subplots(4, 1, figsize = (10, 10), sharex = True)
# axs[0].plot(accel.etime, accel.x, label = "watch x")
# axs[0].plot(accel.etime, accel.y, label = "watch y")
# axs[0].plot(accel.etime, accel.z, label = "watch z")
#
# axs[0].plot(axivity_acc.etime, axivity_acc.x, label = "axivity x")
# axs[0].plot(axivity_acc.etime, axivity_acc.y, label = "axivity y")
# axs[0].plot(axivity_acc.etime, axivity_acc.z, label = "axivity z")
# axs[0].legend()
# axs[0].set_ylabel("Acceleration (g)")
# axs[1].set_ylabel("Acc mag (g)")
# axs[-1].set_xlabel("Time (s)")

# axs[2].plot(pd.to_datetime(green_ppg.etime, unit = "s"), green_ppg.green)
# axs[2].set_ylabel("Green PPG")
# axs[3].plot(pd.to_datetime(red_ppg.etime, unit = "s"), red_ppg.red)
# axs[3].set_ylabel("Red PPG")
# plt.title(f"{user} {device} Accel")
# plt.show()





# Specify the S3 bucket and file path


# file_key = "1/LS2025E/210202054E02/ParsedFiles/230818_114630_Accel.csv"
# local_path = "test_download_s3.csv"
#
# wr.s3.download(path=f"s3://{bucket}/{file_key}", local_file=local_path, boto3_session=session)
# Download the file using awswrangler
# wr.s3.download(path=f"s3://{bucket}/{file_key}", local_file=local_path, session=wr.Session(aws_access_key, aws_secret_key))

# print("File downloaded successfully!")