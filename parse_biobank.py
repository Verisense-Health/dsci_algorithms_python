from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
from datetime import datetime
import linecache
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.dates as md
import scipy
sns.set_style("darkgrid")
import heartpy as hp


def moving_average(signal, window_size):
    """
    Apply a moving average filter to smooth the signal.

    Args:
        signal (ndarray): Input signal to be smoothed.
        window_size (int): Size of the moving window.

    Returns:
        ndarray: Smoothed signal.
    """
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

def minmax_scale(x):
    return ((x - min(x)) / (max(x) - min(x)))

def standard_scale(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

def parse_sample_ppg():
    # SAMPLE 1 green +3986 offset jc
    shimmer = pd.read_csv("/Users/lselig/Desktop/verisense/data/ppg/ppg1/comparison1/shimmerppg1.csv")
    jc = pd.read_csv("/Users/lselig/Desktop/verisense/data/ppg/ppg1/comparison1/greenppg1.csv", skiprows = 8)
    print(jc)
    # jc['millisecond'] = pd.to_datetime(jc['millisecond'])
    jc = jc[(jc.millisecond >= 1686724915024) & (jc.millisecond <= 1686724970502)]

    fs = 25.0  # Sample rate (Hz)
    # duration = 10  # Duration of the signal (seconds)
    # t = np.linspace(0, duration, int(fs * duration))
    f1, f2 = 0.5, 10  # Bandpass frequency range (Hz)
    # ppg_signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    ppg_signal = jc.Unit.values

    # Design the bandpass filter
    order = 4  # Filter order
    nyquist_freq = 0.5 * fs
    low_cutoff = f1 / nyquist_freq
    high_cutoff = f2 / nyquist_freq
    b, a = scipy.signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass')

    # Apply the bandpass filter to the PPG signal
    filtered_ppg = scipy.signal.lfilter(b, a, ppg_signal)

    plt.plot(jc.Unit, label = "orig")
    plt.plot(filtered_ppg[100:], label = "filt")
    plt.legend()
    plt.show()
    # shimmer = shimmer[(shimmer.System_Timestamp >= 1686724915024) & (shimmer.System_Timestamp <= 1686724970502)]
    # sample_rate = (shimmer.shape[0] / ((shimmer.iloc[-1].System_Timestamp - shimmer.iloc[0].System_Timestamp) / 1000))
    # print("SHIMMER sample rate green", sample_rate)
    # wd, m = hp.process(shimmer.F5437a_PPG_A13.values, sample_rate = sample_rate)
    #set large figure
    # plt.figure(figsize=(12,4))
    # plt.plot(minmax_scale(jc.Unit))
    # plt.show()

    # wd, m = hp.process(standard_scale(jc.Unit).values, sample_rate = 25.0)
    wd, m = hp.process(minmax_scale(filtered_ppg[100:]), sample_rate = 25.0)
    #set large figure
    plt.figure(figsize=(12,4))

    #call plotter
    hp.plotter(wd, m)
    plt.show()

    #display measures computed
    for measure in m.keys():
        print('%s: %f' %(measure, m[measure]))


    # sns.set_style("darkgrid")
    # fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)
    # fig.suptitle("JC 2025 vs. Shimmer\nGreen LED")
    # axs[0].set_title("Shimmer")
    # axs[0].set_ylabel("Green PPG")
    # axs[0].plot(shimmer.System_Timestamp, shimmer.F5437a_PPG_A13)
    # # axs[0].plot(shimmer.TimeStamp * 1e6, shimmer.F5437a_PPG_A13)
    # axs[1].set_ylabel("Green PPG")
    # axs[1].set_title("JC2025E")
    # axs[1].plot(jc.millisecond, jc.Unit)

    # axs[2].plot(shimmer.System_Timestamp, minmax_scale(shimmer.F5437a_PPG_A13), label = "0-1 Shimmer Green PPG")
    # axs[2].plot(jc.millisecond, minmax_scale(jc.Unit), label = "0-1 JC2025E Green PPG")
    # axs[2].set_title("Shimmer vs JC2025E")
    # axs[2].set_ylabel("Normalized PPG")
    # axs[2].legend()

    # plt.tight_layout()
    # plt.show()

    # plt.plot(np.diff(shimmer.System_Timestamp))
    # plt.plot(np.diff(jc.millisecond))
    # plt.show()


    # SAMPLE 2 green +3986 offset jc
    shimmer = pd.read_csv("/Users/lselig/Desktop/verisense/data/ppg/ppg1/comparison2/shimmerppg2.csv")
    jc = pd.read_csv("/Users/lselig/Desktop/verisense/data/ppg/ppg1/comparison2/redppg2.csv", skiprows = 8)

    # sample_rate = (shimmer.shape[0] / ((shimmer.iloc[-1].System_Timestamp - shimmer.iloc[0].System_Timestamp) / 1000))
    # print()
    # print("SHIMMER sample rate red", sample_rate)

    # jc = jc[jc.millisecond >= shimmer.System_millisecon]
    # sample_rate = 146.0
    # print("SAMPLE RATE", sample_rate)

    # wd, m = hp.process(shimmer.F5437a_PPG_A13.values, sample_rate = sample_rate)
    #set large figure
    # plt.figure(figsize=(12,4))
    # plt.plot(minmax_scale(jc.Unit))
    # plt.show()
    plt.close()
    plt.plot(minmax_scale(jc.Unit), label = "Scaled JC Red")
    plt.plot(minmax_scale(scipy.signal.detrend(jc.Unit)), label = "Detrended Scaled JC Red")
    plt.legend()
    plt.ylabel("Red PPG")
    plt.title("JC Watch")
    plt.show()

    fs = 100.0  # Sample rate (Hz)
    # duration = 10  # Duration of the signal (seconds)
    # t = np.linspace(0, duration, int(fs * duration))
    f1, f2 = 0.5, 10  # Bandpass frequency range (Hz)
    # ppg_signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    ppg_signal = jc.Unit.values
    plt.close()
    plt.plot(ppg_signal, label = "orig")
    window_size = 10
    ppg_signal = moving_average(ppg_signal, window_size)
    # ppg_signal = ppg_signal[20:-20]
    plt.plot(ppg_signal, label = "smoothed")
    plt.legend()
    plt.title("smoothed")
    plt.show()

    # Design the bandpass filter
    order = 4  # Filter order
    nyquist_freq = 0.5 * fs
    low_cutoff = f1 / nyquist_freq
    high_cutoff = f2 / nyquist_freq
    b, a = scipy.signal.butter(order, [low_cutoff, high_cutoff], btype='bandpass')

    # Apply the bandpass filter to the PPG signal
    filtered_ppg = scipy.signal.lfilter(b, a, ppg_signal)
    plt.plot(filtered_ppg)
    plt.show()

    # wd, m = hp.process(minmax_scale(scipy.signal.detrend(jc.Unit)), sample_rate = 100.0)
    wd, m = hp.process(filtered_ppg[400:], sample_rate = 100.0)

    #call plotter
    hp.plotter(wd, m)


    # print(jc.shape[0], (jc.iloc[-1].millisecond - jc.iloc[0].millisecond))

    # wd, m = hp.process(jc.Unit.values, sample_rate = 100.0)
    #set large figure
    # plt.figure(figsize=(12,4))

    # #call plotter
    # hp.plotter(wd, m)

    # #display measures computed
    # for measure in m.keys():
    #     print('%s: %f' %(measure, m[measure]))


    # print("JC RED SAMPLING RATE", jc.shape[0] / (jc.iloc[-1].millisecond / jc.iloc[0].millisecond) / 100)

    sns.set_style("darkgrid")
    fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)
    print(jc)

    fig.suptitle("JC 2025 vs. Shimmer\nRed LED")
    axs[0].set_title("Shimmer")
    axs[0].set_ylabel("Red PPG")
    axs[0].plot(shimmer.System_Timestamp, shimmer.F5437a_PPG_A13)
    # axs[0].plot(shimmer.TimeStamp * 1e6, shimmer.F5437a_PPG_A13)
    axs[1].set_ylabel("Red PPG")
    axs[1].set_title("JC2025E")
    axs[1].plot(jc.millisecond, jc.Unit)

    axs[2].plot(shimmer.System_Timestamp, minmax_scale(shimmer.F5437a_PPG_A13), label = "0-1 Shimmer Red PPG")
    axs[2].plot(jc.millisecond, minmax_scale(jc.Unit), label = "0-1 JC2025E Red PPG")
    axs[2].set_title("Shimmer vs JC2025E")
    axs[2].set_ylabel("Normalized PPG")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    plt.plot(np.diff(shimmer.System_Timestamp))
    plt.plot(np.diff(jc.millisecond))
    plt.show()


def find_question(colname):
    sago_map = pd.read_csv("/Users/lselig/Desktop/sago_map.csv")
    print(list(sago_map))
    keys = list(sago_map["Unnamed: 1"].values)
    values = list(sago_map["Unnamed: 2"].values)
    keys.append("Q21")
    values.append("How much bodily pain have you had during the past 4 weeks?")

    keys.append("Q22")
    values.append("During the past 4 weeks, how much did pain interfere with your normal work (including both work outside the home and housework)?")

    keys.append("Q32")
    values.append("During the past 4 weeks, how much of the time has your physical health or emotional problems interfered with your social activities (like visiting with friends, relatives, etc.)?")
    for i, key in enumerate(keys):
        try:
            if(colname in key):
                ret = f"{colname}: {values[i]}"
                words = ret.split()
                words.insert(len(words) // 2, "\n")
                return " ".join(words)

        except:
            pass

    return colname


def survey_over_time(survey, subj):
    df = pd.read_csv(survey)
    df = df[df["Device Number"] == subj]
    print(df)
    df = df.sort_values(by = "Survey Month")
    first_half = (4, 24)
    second_half = (24, 40)
    halves = [first_half, second_half]

    for halve in halves:
        start, end = halve[0], halve[1]
        tmp_df = df.iloc[:, start:end]
        fig, axs = plt.subplots(5, 4, figsize = (15, 9))
        axs = axs.flatten()
        for i in range(tmp_df.shape[1]):
            axs[i].plot(range(tmp_df.shape[0]), tmp_df.iloc[:, i], marker = ".")
            axs[i].set_xlabel("Survey Month")
            colname = list(tmp_df)[i]
            axs[i].set_title(find_question(colname))
            axs[i].locator_params(axis='x', integer=True)
        fig.suptitle(f"Survey Over Time By Question\nSubject = {subj}")
        print(tmp_df)
        if(end == 40):
            axs[19].remove()
            axs[18].remove()
            axs[17].remove()
            axs[16].remove()
        plt.tight_layout()
        plt.show()

    # plt.plot(df["Survey Month"], df["Q1"], marker = ".")
    # plt.xlabel("Survey Month")
    # plt.title(f"Survey subject: {subj}")
    # plt.show()
    return df
    # print(df)

def parse_survey(survey, subj):
    df = pd.read_csv(survey)
    df = df[df["Survey Month"] == 1]
    df = df.drop_duplicates(subset = ["Device Number"])
    df = df.sort_values(by = "Survey Month")
    first_half = (4, 24)
    second_half = (24, 40)
    halves = [first_half, second_half]
    # names = ["Q1: In general, would you say your health is:",
    #          "Q2: Compared to one year ago, how would you rate your health in general now?"]

    # my_map = {"first": (["Q1", "Q2", "Q20", "Q21", "Q22", "Q32"], (2, 3), ["In general, would you say your health is:",
    #                                                                       "Compared to one year ago, how would you rate your health in general now?",
    #                                                                       "During the past 4 weeks, to what extent has your physical health or emotional problems interfered with your normal social activities with family, friends, neighbors, or groups?",
    #                                                                       "How much bodily pain have you had during the past 4 weeks?",
    #                                                                       "During the past 4 weeks, how much did pain interfere with your normal work (including both work outside the home and housework)?",
    #                                                                       "During the past 4 weeks, how much of the time has your physical health or emotional problems interfered with your social activities (like visiting with friends, relatives, etc.)?"
    #                                                                       ]),
    #           "second": (["Q3_12r1", "Q3_12r2", "Q3_12r3", "Q3_12r4", "Q3_12r5", "Q3_12r6", "Q3_12r7", "Q3_12r8", "Q3_12r9", "Q3_12r10"], (4, 3), ["The following items are about activities you might do during a typical day. Does your health now limit you in these activities? If so, how much?"]),
    #           "third": (["Q13_16r1", "Q13_16r2", "Q13_16r3", "Q13_16r4"], (2, 2)),
    #           "fourth": (["Q17_19r1", "Q17_19r2", "Q17_19r3"], (2, 2)),
    #           "fifth": (["Q17_19r1", "Q17_19r2", "Q17_19r3"], (2, 2)),
    #          }
    # dims =

    for halve in halves:
        start, end = halve[0], halve[1]
        tmp_df = df.iloc[:, start:end]
        fig, axs = plt.subplots(5, 4, figsize = (15, 9))
        axs = axs.flatten()
        for i in range(tmp_df.shape[1]):
            axs[i].hist(tmp_df.iloc[:, i])
            colname = list(tmp_df)[i]
            axs[i].set_title(find_question(colname))
            axs[i].locator_params(axis='x', integer=True)



        fig.suptitle(f"Hist of Survey By Question\nn = {df.shape[0]}")

        print(tmp_df)
        if(end == 40):
            axs[19].remove()
            axs[18].remove()
            axs[17].remove()
            axs[16].remove()
        plt.tight_layout()
        plt.show()
    return df



# print(parse_survey("/Users/lselig/Desktop/sago_surveys.csv", "C1001-1003"))
# print(survey_over_time("/Users/lselig/Desktop/sago_surveys.csv", "C1001-1006"))
parse_sample_ppg()

def parse_payload(folder):

    headers = ["PayloadIndex", "etime", "End_Timestamp", "Start_Timestamp_Since_Boot", "End_Timestamp_Since_Boot", "Payload_Packaging_Time", "Temperature", "Battery", "PayloadSplitIndex"]
    payloads = glob.glob(f"{folder}ParsedFiles/*Payload*.csv")

    dfs = []
    for p in payloads:
        data = pd.read_csv(p, skiprows = 9)
        dfs.append(data)
        # print(data)

    metadata_df = pd.concat(dfs)
    metadata_df.columns = headers
    metadata_df = metadata_df.sort_values(by = ["etime"])
    print(metadata_df.describe())
    print(metadata_df.head(-5))

    # plt.plot(np.diff(metadata_df.start_timestamp.values))
    # plt.show()

    return metadata_df


def parse_accel_cal(folder):
    headers = ["x", "y", "z", "etime"]
    payloads = glob.glob(f"{folder}ParsedFiles/*Accel_CAL*.csv")
    if(len(payloads) == 0):
        return None

    dfs = []
    for p in payloads:
        if("Archive" in p):
            continue

        try:
            print(p)
            data = pd.read_csv(p, skiprows = 9)
            print(data.describe())
            line = linecache.getline(p, 4)
            print(line.split(";")[-2].split("=")[1][1:])
            start_time = float(line.split(";")[-2].split("=")[1][1:]) * 10 ** -3

            line = linecache.getline(p, 5)
            print(line.split(";")[-2].split("=")[1][1:])
            end_time = float(line.split(";")[-2].split("=")[1][1:]) * 10 ** -3
            print(start_time, end_time)
            # print(end_time - start_time)

            etime = np.linspace(start_time, end_time, num = data.shape[0])

            data["etime"] = etime
            data.columns = headers
            print(data.describe())
            data["mag"] = np.sqrt(data.x ** 2 + data.y ** 2 + data.z ** 2) / 9.8
            dfs.append(data)
            # print(data)
        except:
            continue

    acc_df = pd.concat(dfs)
    acc_df = acc_df.sort_values(by = ["etime"])
    # print(acc_df.describe())
    # print(acc_df.head(-5))

    # plt.plot(np.diff(metadata_df.start_timestamp.values))
    # plt.show()

    return acc_df

# process raw data

# parse_payload("/Users/lselig/Desktop/shimmer/C1001-1004/200728010DBD/")

# users = glob.glob("/Users/lselig/Desktop/shimmer/*/")
# for u in users:
#     print(u)
#     # u = "/Users/lselig/Desktop/shimmer/C1001-1003/"
#     acc_df = parse_accel_cal(f"{u}2007*/")
#     if(acc_df is not None):
#         plt.plot(pd.to_datetime(acc_df.etime, unit = "s"), acc_df.mag)
#         plt.title(u)
#         # plt.show()
#         plt.savefig(f"/Users/lselig/Desktop/shimmer_parsed/{u.split('/')[-1]}.png")
#         plt.close()
#         acc_df.to_csv(f"/Users/lselig/Desktop/shimmer_parsed/{u.split('/')[-1]}.csv", index = False)


# process ggir summary data
def split_colname(c):

    ret = []
    for i, v in enumerate(c):
        if(v == "_"):
            ret.append(i)

    if(len(ret) == 0):
        return c

    mid = ret[len(ret) // 2]
    c = list(c)
    c[mid] = "\n"
    c = "".join(c)
    return c





def do_pca(data, subj):
    print(data)
    data = data.drop(columns = ["id", "yyyy-MM-dd", "weekday", "date", "L5TIME", "M5TIME", "my_datetime"])
    print(list(data))
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    print(pca_data)


    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker = ".")
    plt.axhline(0.8, ls = "--", c = "red")
    plt.xlabel("# of principal components")
    plt.ylabel("% variance explained")
    plt.title(f"PCA variance explained\n{data.shape}")
    plt.ylim(0, 1)
    # plt.savefig(f"/Users/lselig/Desktop/PCA_{subj}_var_ex.png", dpi = 500)
    plt.close()
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    print(pca_data)

    # Create a new dataframe with the PCA results
    pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])

    # Print the explained variance ratio
    print('Explained Variance Ratio:', pca.explained_variance_ratio_)

    # Print the transformed data
    print('Transformed Data:')
    print(pca_df)

    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title(f"PCA\nSubject: {subj}\nExplained Variance: {pca.explained_variance_ratio_}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # plt.savefig(f"/Users/lselig/Desktop/PCA_{subj}.png", dpi = 500)
    plt.close()
    # plt.show()
    return pca_df

def main():
    total_days = 0
    dfs = []

    t_tests = []
    ks_tests = []

    for s in glob.glob("/Users/lselig/Desktop/shimmer/*/Algorithms/GGIR_Summary/*Activity_Summary.csv"):
        subj = s.split("/")[-4]
        df = pd.read_csv(s)
        df["date"] = [datetime.strptime(x[0], "%Y-%m-%d").date() for x in df[["yyyy-MM-dd"]].values]

        # if(subj != "C1001-1016"):
        #     continue
        # if(subj == "C1001-1016"):
        #     df.drop(df.loc[df['yyyy-MM-dd']=="2023-03-10"].index, inplace=True)

        df["my_datetime"] = pd.to_datetime(df.date)
        dfs.append(df)
        weekends = df[(df.weekday == "Saturday") | (df.weekday == "Sunday")]
        weekdays = df[(df.weekday != "Saturday") & (df.weekday != "Sunday")]
        features = ["dur_day_total_IN_min",
                    "dur_day_total_LIG_min",
                    "dur_day_total_MVPA_min",
                    "M5VALUE",
                    "L5VALUE"]

        for f in features:
            n_weekdays = weekdays.shape[0]
            n_weekends = weekends.shape[0]
            pvalue_ttest = stats.ttest_ind(weekends[f].values, weekdays[f].values).pvalue
            results_ttest = pd.DataFrame([{"n_weekdays": n_weekdays,
                                        "n_weekends": n_weekends,
                                        "pvalue": pvalue_ttest,
                                        "feature": f,
                                        "individual": subj}])

            if(weekdays.shape[0] == 0 or weekends.shape[0] == 0):
                pvalue_ks = np.nan
            else:
                pvalue_ks = stats.ks_2samp(weekends[f].values, weekdays[f].values).pvalue
            results_kstest = pd.DataFrame([{"n_weekdays": n_weekdays,
                                        "n_weekends": n_weekends,
                                        "pvalue": pvalue_ks,
                                        "feature": f,
                                        "individual": subj}])

            t_tests.append(results_ttest)
            ks_tests.append(results_kstest)


        print(t_tests)
        big_df = df
        date_diff = df['my_datetime'].diff()
        start = 0
        end = 0
        chunks = []
        for i, d in enumerate(date_diff):
            if(d == pd.Timedelta(days = 1)):
                end += 1
                continue
            else:
                chunks.append(df[start:end])
                start = end
                end += 1

        # consecutive_dates = df[date_diff == pd.Timedelta(days=1)]
        # df = consecutive_dates
        total_days += df.shape[0]

        chunks.append(df[date_diff != pd.Timedelta(days = 1)])
        do_pca(df, subj = s.split("/")[-4])
        # dfs.append(df)
        # continue
        columns = ["dur_day_total_IN_min", "dur_day_total_LIG_min", "dur_day_total_MVPA_min", "M5VALUE", "L5VALUE"]
        fig, axs = plt.subplots(len(columns), 1, figsize = (15, 9), sharex = True)

        axs[-1].set_xlabel("Time")

        for i, c in enumerate(columns):

            for j, chunk in enumerate(chunks):
                df = chunk
                mean = np.nanmean(big_df[[c]])
                std = np.nanstd(big_df[[c]])
                if(j == len(chunks) -1):
                    axs[i].scatter(df.date, df[[c]], marker = ".", color = f"C{i}", zorder = 2)
                else:
                    axs[i].plot(df.date, df[[c]], marker = ".", color = f"C{i}", zorder = 2)
                # print(c)
                # print(df)
                # print(mean + std)
                above = df[df[[c]].values >= mean + std]
                axs[i].scatter(above.date, above[[c]], color = "red", s = 50, marker = "X", zorder = 3)

                below = df[df[[c]].values <= mean - std]
                axs[i].scatter(below.date, below[[c]], color = "red", s = 50, marker = "X", zorder = 3)

                if(j == 0):
                    axs[i].axhline(mean, alpha = 0.5, color = "black", label = "Mean" if i == 0 else None, ls = "--", zorder = 1)
                    axs[i].axhline(mean + std, alpha = 0.5, color = "red", label = "1 Std" if i == 0 else None, ls = "--", zorder = 1)
                    axs[i].axhline(mean - std, alpha = 0.5, color = "red", ls = "--", zorder = 1)
                ylabel = split_colname(c)
                axs[i].set_ylabel(ylabel)
                xfmt = md.DateFormatter('%Y-%m-%d\n%a')
                axs[i].xaxis.set_major_formatter(xfmt)
                fig.legend()

        # handles, labels = fig.get_legend_handles_labels()
        # unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        # fig.legend(*zip(*unique))


        algorithm = "/".join(s.split("/")[-3:])
        ndays = big_df.shape[0]
        fig.suptitle(f"Subject: {subj}\nAlgorithm: {algorithm}\nN days: {ndays}")

        # plt.savefig(f"/Users/lselig/Desktop/{subj}.png", dpi = 500)
        plt.close()
        # plt.show()

    big_ttest = pd.concat(t_tests)
    big_ttest.sort_values(by = ["pvalue"])
    big_ttest.to_csv("/Users/lselig/Desktop/all_ttests.csv", index = False)

    big_kstest = pd.concat(ks_tests)
    big_kstest.sort_values(by = ["pvalue"])
    big_kstest.to_csv("/Users/lselig/Desktop/all_kstests.csv", index = False)

    # big_df = pd.concat(dfs)



    print(f"N days of data: {total_days}")
    big_df = pd.concat(dfs)
    big_df.to_csv("/Users/lselig/Desktop/0140_shimmer_activity_summary.csv", index = False)
    return big_df







ks_test = pd.read_csv("/Users/lselig/Desktop/all_kstests.csv")
ts_test = pd.read_csv("/Users/lselig/Desktop/all_ttests.csv")
alpha = 0.05
ks_test = ks_test[ks_test.pvalue <= alpha]
ts_test = ts_test[ts_test.pvalue <= alpha]
merged = pd.merge(ks_test, ts_test, how='inner', on=["feature", "individual"])

print(ks_test)
print(ts_test)
df = main()
corr_matrix = df.corr()
corr_matrix["average"] = corr_matrix.mean(axis = 1)
print(corr_matrix)

# Create a heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"fontsize": 8})
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)

# Set the title and display the plot
plt.title('Correlation Matrix\nFor all days for all subjects')
plt.show()
weekends = df[(df.weekday == "Saturday") | (df.weekday == "Sunday")]
weekdays = df[(df.weekday != "Saturday") & (df.weekday != "Sunday")]
features = ["dur_day_total_IN_min",
        "dur_day_total_LIG_min",
        "dur_day_total_MVPA_min",
        "M5VALUE",
        "L5VALUE"]

big_ttest, big_kstest = [], []
for f in features:
    n_weekdays = weekdays.shape[0]
    n_weekends = weekends.shape[0]
    pvalue_ttest = stats.ttest_ind(weekends[f].values, weekdays[f].values).pvalue
    results_ttest = pd.DataFrame([{"n_weekdays": n_weekdays,
                                "n_weekends": n_weekends,
                                "pvalue": pvalue_ttest,
                                "feature": f}])

    if(weekdays.shape[0] == 0 or weekends.shape[0] == 0):
        pvalue_ks = np.nan
    else:
        pvalue_ks = stats.ks_2samp(weekends[f].values, weekdays[f].values).pvalue
    results_kstest = pd.DataFrame([{"n_weekdays": n_weekdays,
                                "n_weekends": n_weekends,
                                "pvalue": pvalue_ks,
                                "feature": f}])

    big_ttest.append(results_ttest)
    big_kstest.append(results_kstest)

print(len(big_ttest), len(big_kstest))
pd.concat(big_ttest).to_csv("/Users/lselig/Desktop/ttest_population.csv", index = False)
pd.concat(big_kstest).to_csv("/Users/lselig/Desktop/kstest_population.csv", index = False)


