from collections import OrderedDict
import numpy as np
import pandas as pd
import dateutil
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style("darkgrid")

# write a function to read the 4th line of a csv file
def read_line(infile, linei):
    with open(infile, "r") as f:
        for i, line in enumerate(f):
            if(i == linei - 1):
                return line
    return None

# 9:35 - 10:30 start
# 9:41 - 10:36 end
# face up
# face down
# watch facing right
def parse_imu(infile):
    line4 = read_line(infile, 4)
    line4 = line4.split("Local time zone offset = ")
    start = float(line4[1].split(";")[0]) * 1e-3

    line5 = read_line(infile, 5)
    line5 = line5.split("Local time zone offset = ")
    end = float(line5[1].split(";")[0]) * 1e-3

    df = pd.read_csv(infile, skiprows = 9)
    # write code to rename columns of a dataframe
    df = df.rename(columns = {"m/(s^2)": "x", "m/(s^2).1": "y", "m/(s^2).2": "z"})
    df["x"] = df["x"] / 9.81
    df["y"] = df["y"] / 9.81
    df["z"] = df["z"] / 9.81

    n = df.shape[0]
    df["etime"] = np.linspace(start, end, n)
    print(df)
    print(df.describe())
    return df


def parse_2025e(infile):
    df = pd.read_csv(infile)
    fname = "First Test"
    timestamps = []
    etimes = []
    samples = []
    timestamps_plot = []
    for i,ts in enumerate(df["Date"]):
        if(i == 0):
            end = int(dateutil.parser.parse(df.iloc[0]["Date"]).strftime("%s")) - 1
        else:
            end = int(dateutil.parser.parse(df.iloc[i - 1]["Date"]).strftime("%s"))

        start = int(dateutil.parser.parse(df.iloc[i]["Date"]).strftime("%s"))
        etimes.append(start)
        timestamps_plot.append(start)
        timestamps += list(np.linspace(start, end, 28))
        for j in range(1, 29):
            samp = [df["X" + str(j)].values[i], df["Y" + str(j)].values[i], df["Z" + str(j)].values[i]]
            samples.append(samp)



    df = pd.DataFrame({"x": np.array([s[0] for s in samples]) / 256, "y": np.array([s[1] for s in samples]) / 256, "z": np.array([s[2] for s in samples]) / 256})
    # df = df.dot(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]))
    df = df.rename(columns = {0: "x", 1: "y", 2: "z"})
    df["etime"]  = np.array(timestamps) + 3600 * -5
    df = df.sort_values(by = "etime")

    return df


def parse_axivity(infile, show_plot):
    df = pd.read_csv(infile)
    df.columns = ["etime", "x", "y", "z"]
    if(show_plot):
        fig, axs = plt.subplots(3, 1, figsize = (15, 10), sharex = True)
        plt.plot(pd.to_datetime(df.etime, unit = "s"), df.x, color = "r")
        plt.plot(pd.to_datetime(df.etime, unit = "s"), df.y, color = "g")
        plt.plot(pd.to_datetime(df.etime, unit = "s"), df.z, color = "b")
        plt.show()
    return df
# baseline1 = [1690837507, 1690837622]
# claps1 = [1690837622, 1690837653]
# walk1 = [1690837656, 1690837784]
# claps2 = [1690837787, 1690837814]
# baseline2 = [1690837815, 1690837925]
#
# laps = [baseline1, claps1, walk1, claps2, baseline2]
# labels = ["Baseline", "Claps", "Walk", "Claps", "Baseline"]


face_up = [1690903800, 1690903872]
face_down = [1690903872, 1690903930]
face_right = [1690903930, 1690903988]
face_left = [1690903988, 1690904049]
face_towards = [1690904049, 1690904049 + 60]
face_away = [1690904049 + 60, 1690904049 + 60 + 60]

laps = [face_up, face_down, face_right, face_left, face_towards, face_away]
labels = ["Face Up", "Face Down", "Face Right", "Face Left", "Face Towards", "Face Away"]


# jcw_df = parse_2025e("/Users/lselig/Desktop/test_gsensor_1.csv")
# jcw_df.to_csv("/Users/lselig/Desktop/test_gsensor_1_parsed.csv", index = False)
# imu_df = parse_imu("/Users/lselig/Desktop/230731_143255_Accel_DEFAULT_CAL_00607.csv")
# imu_df["etime"] = imu_df["etime"] - 338

def plot_indv(df, tz, laps, labels, title, do_baseline_hist):

    naxes = 5
    if(not do_baseline_hist):
        fig, axs = plt.subplots(naxes, 1, sharex = True, figsize = (15, 9))
    baseline_dfs = []

    for i, lap in enumerate(laps):
        if (labels[i] == "Baseline" and len(baseline_dfs) == 0):
            color = "pink"
            # sns.set_style("darkgrid")
            tmp = df[df.etime.between(lap[0] + 5 + 3600 * tz, lap[1] - 5 + 3600 * tz)]
            tmp["mag"] =np.sqrt(tmp.x ** 2 + tmp.y ** 2 + tmp.z ** 2)
            baseline_dfs.append(tmp)
        elif(labels[i] == "Claps"):
            color = "purple"
        else:
            color = f"C{i+2}"

        for j in range(naxes):
            if(j == 4 and not do_baseline_hist):
                axs[j].axvspan(pd.to_datetime(laps[i][0] + 3600 * tz, unit = "s"), pd.to_datetime(laps[i][1] + 3600 * tz, unit = "s"), facecolor = color, alpha = 0.3, zorder = 3)
            elif(not do_baseline_hist):
                axs[j].axvspan(pd.to_datetime(laps[i][0] + 3600 * tz, unit = "s"), pd.to_datetime(laps[i][1] + 3600 * tz, unit = "s"), facecolor = color, alpha = 0.3, zorder = 3, label = labels[i])

    # baseline_df = pd.concat(baseline_dfs)
    # return baseline_df

    # plt.plot(tmp.etime, np.sqrt(tmp.x ** 2 + tmp.y ** 2 + tmp.z ** 2))
    # plt.show()

    by_label = OrderedDict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(),
                  loc='upper center', bbox_to_anchor=(0.5, 1.50),
                      ncol=3, fancybox=True, shadow=True)


    fig.suptitle(title)
    axs[0].plot(pd.to_datetime(df.etime, unit = "s"), df.x.values, label = "x", color = "C0")
    axs[0].set_ylabel("X(g)")
    axs[1].plot(pd.to_datetime(df.etime, unit = "s"), df.y.values, label = "y", color = "C1")
    axs[1].set_ylabel("Y(g)")
    axs[2].plot(pd.to_datetime(df.etime, unit = "s"), df.z.values, label = "z", color = "C2")
    axs[2].set_ylabel("Z(g)")
    axs[3].plot(pd.to_datetime(df.etime, unit = "s"), np.sqrt(df.x.values ** 2 + df.y.values ** 2 + df.z.values ** 2), label = "mag", color = "black")
    axs[3].set_ylabel("Magnitude(g)")
    axs[-1].set_xlabel("Time(s)")
    axs[4].plot(pd.to_datetime(df.etime, unit = "s"), df.x.values, label = "x", color = "C1", alpha = 1.0)
    axs[4].plot(pd.to_datetime(df.etime, unit = "s"), df.y.values, label = "y", color = "C2", alpha = 1.0)
    axs[4].plot(pd.to_datetime(df.etime, unit = "s"), df.z.values, label = "z", color = "C3", alpha = 1.0)
    axs[4].legend()
    # plt.tight_layout()
    plt.show()
def plot_compare(df1, df2, tz, laps, labels, title):

    df1 = df1[(df1.etime >= laps[0][0] + 3600 * tz) & (df1.etime <= laps[-1][1] + 3600 * tz)]
    df2 = df2[(df2.etime + 3600 >= laps[0][0] + 3600 * tz) & (df2.etime + 3600 <= laps[-1][1] + 3600 * tz)]
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 9))
    for i, lap in enumerate(laps):
        if (labels[i] == "Baseline"):
            color = "pink"
        elif (labels[i] == "Claps"):
            color = "purple"
        else:
            color = f"C{i+2}"

        for j in range(len(axs)):
            axs[j].set_ylim(-1.2, 1.2)
            if (j == 3):
                axs[j].axvspan(pd.to_datetime(laps[i][0] + 3600 * tz, unit="s"),
                               pd.to_datetime(laps[i][1] + 3600 * tz, unit="s"), facecolor=color, alpha=0.3, zorder=3)
            else:
                axs[j].axvspan(pd.to_datetime(laps[i][0] + 3600 * tz, unit="s"),
                               pd.to_datetime(laps[i][1] + 3600 * tz, unit="s"), facecolor=color, alpha=0.3, zorder=3,
                               label=labels[i])


    fig.suptitle(title)
    axs[0].plot(pd.to_datetime(df1.etime, unit="s"), df1.x.values, label="Smartwatch", color="C0")
    # tw = axs[0].twinx()
    # axs[0].plot(pd.to_datetime(df1.iloc[0].etime, unit="s"), np.nan, label="IMU", color="C1")
    axs[0].plot(pd.to_datetime(df2.etime + 3600 , unit="s"), df2.x.values, label="IMU", color="C1")
    axs[0].set_ylabel("X(g)")
    # tw.set_ylabel("IMU: X(g)")
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    axs[1].plot(pd.to_datetime(df1.etime, unit="s"), df1.y.values, label="Smartwatch Y", color="C0")
    # tw = axs[1].twinx()
    axs[1].plot(pd.to_datetime(df2.etime + 3600, unit="s"), df2.y.values, label="IMU Y", color="C1")
    axs[1].set_ylabel("Y(g)")
    # tw.set_ylabel("IMU: Y(g)")

    axs[2].plot(pd.to_datetime(df1.etime, unit="s"), df1.z.values, label="Smartwatch Z", color="C0")
    # tw = axs[2].twinx()
    axs[2].plot(pd.to_datetime(df2.etime + 3600, unit="s"), df2.z.values, label="IMU Z", color="C1")
    axs[2].set_ylabel("Z(g)")
    # tw.set_ylabel("IMU: Z(g)")

    axs[3].plot(pd.to_datetime(df1.etime, unit="s"), np.sqrt(df1.x.values ** 2 + df1.y.values ** 2 + df1.z.values ** 2),
                label="Smartwatch Mag", color="C0")
    # tw = axs[3].twinx()

    axs[3].plot(pd.to_datetime(df2.etime + 3600, unit="s"), np.sqrt(df2.x.values ** 2 + df2.y.values ** 2 + df2.z.values ** 2),
                label="IMU Mag", color="C1")

    # axs[3].plot(pd.to_datetime(df.etime, unit="s"), np.sqrt(df.x.values ** 2 + df.y.values ** 2 + df.z.values ** 2),
    #             label="mag", color="black")
    axs[3].set_ylabel("Magnitude(g)")
    # tw.set_ylabel("IMU: Magnitude(g)")

    axs[-1].set_xlabel("Time(s)")
    # axs[4].plot(pd.to_datetime(df.etime, unit="s"), df.x.values, label="x", color="C1", alpha=1.0)
    # axs[4].plot(pd.to_datetime(df.etime, unit="s"), df.y.values, label="y", color="C2", alpha=1.0)
    # axs[4].plot(pd.to_datetime(df.etime, unit="s"), df.z.values, label="z", color="C3", alpha=1.0)
    # axs[4].legend()
    # plt.tight_layout()
    axs[0].legend(by_label.values(), by_label.keys(),
                  loc='upper center', bbox_to_anchor=(0.5, 1.50),
                  ncol=3, fancybox=True, shadow=True)
    plt.show()


jcw_df = parse_2025e("/Users/lselig/Desktop/test_gsensor_lucas_0817.csv")
axv_df = parse_axivity("/Users/lselig/Desktop/test_axivitiy_lucas_0817.csv", show_plot = True)
print(jcw_df.shape[0])
print(axv_df.shape[0])
fig, axs = plt.subplots(4, 1, figsize = (15, 9), sharex = True)
axs[0].plot(pd.to_datetime(jcw_df.etime, unit = "s"), jcw_df.x)
axs[0].plot(pd.to_datetime(axv_df.etime, unit = 's'), axv_df.x)
plt.show()
# plot_compare(jcw_df, axv_df, -5, )

jcw_df = parse_2025e("/Users/lselig/Desktop/test_gsensor_orientation.csv")
jcw_df.to_csv("/Users/lselig/Desktop/test_gsensor_1_parsed.csv", index = False)
imu_df = parse_imu("/Users/lselig/Desktop/test_imu_orientation.csv")
imu_df["etime"] = imu_df["etime"] - 335
plot_compare(jcw_df, imu_df, -5, laps, labels, title = "Protocol")
# jcw_baseline_df = plot_indv(jcw_df, -5, laps, labels, "Verisense SmartWatch", do_baseline_hist=False)
# imu_baseline_df = plot_indv(imu_df, -6, laps, labels, "Verisense IMU", do_baseline_hist=False)
# plt.hist(jcw_baseline_df.mag, bins = 50, alpha = 0.7)
# plt.title("Baseline noise: Verisense Smartwatch")
# plt.ylabel("Count")
# plt.xlabel("Magnitude (g)")
# median = np.nanmedian(jcw_baseline_df.mag)
# _std = np.nanstd(jcw_baseline_df.mag)
# print(median, std)

# plt.axvline(x = median, label = f"Median: {median: .2f}", color = "red", zorder = 3)
# plt.axvline(x = median + _std, label = f"Sigma: {_std: .3f}", color = "black", ls = "--", zorder = 3)
# plt.axvline(x = median - _std,  color = "black", ls = "--", zorder = 3)
# plt.legend()
# plt.show()
# plt.plot(jcw_baseline_df.etime, jcw_baseline_df.x, label = "Smartwatch")
# plt.plot(jcw_baseline_df.etime, jcw_baseline_df.y, label = "Smartwatch")
# plt.plot(jcw_baseline_df.etime, jcw_baseline_df.z, label = "Smartwatch")
# plt.show()

# fig, axs = plt.subplots(1, 4, figsize = (15, 9))
# nbins = 10
# axs[0].hist(jcw_baseline_df.x,  bins = nbins, alpha = 0.5, label = "Smartwatch")
# axs[0].hist(imu_baseline_df.x,  bins = nbins, alpha = 0.5, label = "IMU")
# axs[0].set_xlabel("X (g)")
#
# axs[1].hist(jcw_baseline_df.y,  bins = nbins, alpha = 0.5, label = "Smartwatch")
# axs[1].hist(imu_baseline_df.y,  bins = nbins, alpha = 0.5, label = "IMU")
# axs[1].set_xlabel("Y (g)")
#
# axs[2].hist(jcw_baseline_df.z,  bins = nbins, alpha = 0.5, label = "Smartwatch")
# axs[2].hist(imu_baseline_df.z,  bins = nbins, alpha = 0.5, label = "IMU")
# axs[2].set_xlabel("Z (g)")
#
# axs[3].hist(jcw_baseline_df.mag,  bins = nbins, alpha = 0.5, label = "Smartwatch")
# axs[3].hist(imu_baseline_df.mag,  bins = nbins, alpha = 0.5, label = "IMU")
# axs[3].set_xlabel("Mag (g)")
# handles, labels = axs[0].get_legend_handles_labels()
# by_label = OrderedDict(zip(labels, handles))
# axs[0].legend(by_label.values(), by_label.keys(),
#               loc='upper center', bbox_to_anchor=(0.5, 1.50),
#                   ncol=3, fancybox=True, shadow=True)
#
#
# plt.hist(baseline_df.mag, density = True, bins = 40)
# fig.suptitle(f"Baseline Noise: Compare")
# for i in range(4):
#     axs[i].set_ylabel("Count")
# plt.show()
# plot_compare(jcw_df, imu_df, -5, laps, labels, title = "Orientation")
# plot_compare(jcw_df, imu_df, -5, laps, labels, title = "Test1")
