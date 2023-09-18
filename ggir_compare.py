import pandas as pd
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from dsci_tools import replace_gaps
sns.set_theme(style="darkgrid")

def compare_sleep(USER, DEVICE, verisense_sleep_df, axivity_sleep_df):

    verisense_sleep_df["device"] = "Verisense"
    axivity_sleep_df["device"] = "Axivity"

    tz_offset = 5
    sleep_df = pd.concat([verisense_sleep_df, axivity_sleep_df])

    my_truth = pd.DataFrame({"night": [1, 2, 3],
                             "wakeup": [26.8, 25.42, 25.39],
                             "sleeponset": [19.07, 19.14, 18.4],
                             "device": ["Journal"] * 3})
    sleep_df = pd.concat([sleep_df, my_truth])
    sleep_df["sleeponset"] = sleep_df["sleeponset"] + tz_offset
    sleep_df["wakeup"] = sleep_df["wakeup"] + tz_offset

    titles = ["Detected waking time (after sleep period) expressed\n as hours since the midnight of the previous night.",
              "Detected onset of sleep expressed as hours since\n the midnight of the previous night.",
              "Difference between onset and waking time.",
              "Total sleep duration, which equals the accumulated nocturnal\n sustained inactivity bouts within the Sleep Period Time.",
              "The Sleep Regularity Index as proposed by Phillips et al. 2017, but calculated per day-pair to enable user to study patterns across days",
              "Fraction of the night (noon-noon or 6pm-6pm) for which the data was invalid, e.g. monitor not worn or no accelerometer measurement started/ended within the night."]

    features = ["wakeup", "sleeponset", "SptDuration", "SleepDurationInSpt", "SleepRegularityIndex", "fraction_night_invalid"]
    for i, feature in enumerate(features[:4]):
        g = sns.catplot(
            data=sleep_df, kind="bar",
            x="night", y=feature, hue="device",
            errorbar="sd", palette="dark", alpha=.6, height=6
        )
        g.despine(left=True)
        g.set_axis_labels("", feature)
        g.legend.set_title("")
        plt.title(titles[i])
        plt.xlabel("Night")
        plt.tight_layout()
        plt.savefig(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/comparisons/sleep/{USER}_{DEVICE}_{feature}_sleep_compare.png", dpi = 200)
        plt.show()


def compare_activity_summary(USER, DEVICE, verisense_activity_df, axivity_activity_df):
    features = ["dur_day_total_IN_min", "dur_day_total_LIG_min", "dur_day_total_MOD_min", "dur_day_total_VIG_min",
                "dur_day_min", "L5VALUE", "M5VALUE", "nonwear_perc_day"]
    fig, axs = plt.subplots(2, 4, figsize = (15, 9))
    axs = axs.flatten()
    for i, feature in enumerate(features):
        axs[i].plot(verisense_activity_df[feature], marker = "x", label = "Verisense")
        axs[i].plot(axivity_activity_df[feature], marker = "x", label = "Axivity")
        axs[i].legend()
        axs[i].set_title(feature)
        if("min" in feature):
            axs[i].set_ylabel("Minutes")
        else:
            axs[i].set_ylabel(feature)
        axs[i].set_xlabel("Day")

    fig.suptitle("Lucas Longitudinal Data: Verisense vs. Axivity")
    plt.tight_layout()
    plt.show()

def debug_ggir(verisense_infile, reference_infile, v_acc, a_acc):
    v_acc = pd.read_csv(v_acc)
    a_acc = pd.read_csv(a_acc)

    v_acc = replace_gaps(v_acc, show_plot = False)
    a_acc = replace_gaps(a_acc, show_plot=False)

    start = max(v_acc.iloc[0].etime, a_acc.iloc[0].etime)
    end = min(v_acc.iloc[-1].etime, a_acc.iloc[-1].etime)

    v_acc = v_acc[v_acc.etime.between(start, end)]
    a_acc = a_acc[a_acc.etime.between(start, end)]
    v_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_clean/verisense_acc.csv", index = False)
    a_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity_clean/axivity_acc.csv", index = False)
    plt.close()
    plt.plot(np.diff(v_acc.etime.values))
    plt.show()
    v_acc["mag"] = np.sqrt(v_acc.x ** 2 + v_acc.y ** 2 + v_acc.z ** 2)
    a_acc["mag"] = np.sqrt(a_acc.x ** 2 + a_acc.y ** 2 + a_acc.z ** 2)
    verisense_df = pd.read_csv(verisense_infile)
    reference_df = pd.read_csv(reference_infile)
    v_etime = [parser.parse(x) for x in verisense_df.etime.values]
    r_etime = [parser.parse(x) for x in reference_df.etime.values]
    fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)

    axs[0].plot(v_etime, verisense_df.mag, label = "verisense")
    axs[0].plot(r_etime, reference_df.mag, label = "axivity")
    axs[0].legend()
    axs[0].set_ylabel("ENMO (g)")
    axs[0].set_xlabel("Time (s)")
    axs[1].plot(pd.to_datetime(v_acc.etime, unit = "s"), v_acc.mag)
    axs[2].plot(pd.to_datetime(a_acc.etime, unit = "s"), a_acc.mag)

    plt.show()


def compare_activity_raw(USER, DEVICE, RUN_VERSION, SUFFIX):
    verisense_activity_df = pd.read_csv(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_2025E_{RUN_VERSION}/output_ggir_inputs_2025E_{SUFFIX}/meta/ms5.outraw/40_100_400/verisense_acc_T5A5.csv")
    axivity_activity_df = pd.read_csv(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_axivity_{RUN_VERSION}/output_ggir_inputs_axivity_{SUFFIX}/meta/ms5.outraw/40_100_400/axivity_acc_T5A5.csv")
    naxis = 2
    fig, axs = plt.subplots(naxis, 1, figsize = (15, 9))
    axs[0].plot(verisense_activity_df.timenum, verisense_activity_df.ACC, label = "Verisense", color = "C0")
    axs[0].set_ylabel("Window ENMO")
    axs[0].set_title("Verisense")
    axs[1].plot(axivity_activity_df.timenum, axivity_activity_df.ACC, label = "Axivity", color = "C1")
    axs[1].set_ylabel("Window ENMO")
    axs[1].set_title("Axivity")
    for i in range(naxis):
        axs[i].axhline(30, color = "green", ls = "--", alpha = 0.5, label = "Light")
        axs[i].axhline(100, color = "orange", ls = "--", alpha = 0.5, label = "Moderate")
        axs[i].axhline(400, color = "red", ls = "--", alpha = 0.5, label = "Vigorous")
        axs[i].legend()
    plt.legend()
    fig.suptitle("GGIR Compare - Lucas Longitudinal Data")
    plt.show()
    #
    plt.plot(verisense_activity_df.timenum, verisense_activity_df.ACC, label = "Verisense", color = "C0")
    plt.plot(axivity_activity_df.timenum, axivity_activity_df.ACC, label = "Axivity", color = "C1")
    plt.legend()
    plt.xlabel("Window")
    plt.ylabel("Window ENMO")
    plt.show()



if __name__ == "__main__":
    # v_infile_ggir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v5/output_ggir_inputs_2025E/meta/basic/verisense_ggir_metrics.csv"
    # a_infile_ggir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v5/output_ggir_inputs_axivity/meta/basic/axivity_ggir_metrics.csv"

    # v = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v5/output_ggir_inputs_2025E_clean/meta/ms2.out/verisense_acc.csv")
    # a = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v5/output_ggir_inputs_axivity_clean/meta/ms2.out/axivity_acc.csv")

    # fig, axs = plt.subplots(2, 1, sharex = True, figsize = (15, 9))
    # axs[0].plot([parser.parse(x) for x in v.etime], v.mag, label = "Verisense")
    # axs[0].plot([parser.parse(x) for x in a.etime], a.mag, label = "Axivity")
    # axs[0].set_ylabel("ENMO (g)")
    # axs[0].legend()
    #
    # axs[1].plot([parser.parse(x) for x in v.etime], v.anglez, label = "Verisense")
    # axs[1].plot([parser.parse(x) for x in a.etime], a.anglez, label = "Axivity")
    # axs[1].set_ylabel("Angle (deg)")
    # axs[1].legend()
    # plt.show()



    # v_acc = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_clean/verisense_acc.csv"
    # a_acc = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity_clean/axivity_acc.csv"
    # v_acc = pd.read_csv(v_acc)
    # a_acc = pd.read_csv(a_acc)
    #
    # plt.plot(pd.to_datetime(v_acc.etime, unit = "s"), np.sqrt(v_acc.x ** 2 + v_acc.y ** 2 + v_acc.z ** 2), label = "verisense", alpha = 0.6)
    # plt.plot(pd.to_datetime(a_acc.etime, unit = "s"), np.sqrt(a_acc.x ** 2 + a_acc.y ** 2 + a_acc.z ** 2), label = "axivity", alpha = 0.6)
    # plt.legend()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Magnitude of Acc(g)")
    # plt.title("Lucas Longitudinal Acceleration Data")
    # plt.show()
    # debug_ggir(v_infile_ggir,
    #           a_infile_ggir,
    #           v_acc,
    #           a_acc)
    USER = "LS2025E"
    DEVICE = "210202054E02"
    RUN_VERSION = "longitudinal"
    SUFFIX = "longitudinal"
    stopwatch_df = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/activity_tracker.csv")
    verisense_activity_df = pd.read_csv(
        f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_2025E_{RUN_VERSION}/output_ggir_inputs_2025E_{SUFFIX}/meta/ms5.outraw/40_100_400/verisense_acc_T5A5.csv")
    raw_data = pd.read_csv(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_inputs/ggir_inputs_2025E_{RUN_VERSION}/verisense_acc.csv")
    naxis = 2

    fig, axs = plt.subplots(naxis, 1, figsize=(15, 9))
    axs[0].plot(raw_data.etime, np.sqrt(raw_data.x ** 2 + raw_data.y ** 2 + raw_data.z ** 2) - 1)
    axs[0].set_ylabel("ENMO")

    axs[1].plot(verisense_activity_df.timenum, verisense_activity_df.ACC, label="Verisense", color="C0")
    axs[1].set_ylabel("GGIR Window ENMO")
    axs[1].set_title("Verisense")
    activity_colors = {}
    activities = np.unique(stopwatch_df.activity.values)
    for i, activity in enumerate(activities):
        activity_colors[activity] = f"C{i}"

    for i in range(naxis):
        for label in stopwatch_df.itertuples():
            a = 1
            axs[i].axvspan(label.start, label.stop, alpha=0.5, color = activity_colors[label.activity], label = label.activity)


        axs[i].axhline(30, color="green", ls="--", alpha=0.5, label="Light")
        axs[i].axhline(100, color="orange", ls="--", alpha=0.5, label="Moderate")
        axs[i].axhline(400, color="red", ls="--", alpha=0.5, label="Vigorous")
        axs[i].legend()

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys(),
                  loc='upper center', bbox_to_anchor=(0.5, 1.50),
                  ncol=3, fancybox=True, shadow=True)
    fig.suptitle("GGIR Compare - Lucas Longitudinal Data")
    plt.show()

    USER = "LS2025E"
    DEVICE = "210202054E02"
    RUN_VERSION = "v5"
    SUFFIX = "clean"
    verisense_sleep_df = pd.read_csv(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_2025E_{RUN_VERSION}/output_ggir_inputs_2025E_{SUFFIX}/results/part4_nightsummary_sleep_cleaned.csv")
    axivity_sleep_df = pd.read_csv(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_axivity_{RUN_VERSION}/output_ggir_inputs_axivity_{SUFFIX}/results/part4_nightsummary_sleep_cleaned.csv")

    verisense_activity_df = pd.read_csv(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_2025E_{RUN_VERSION}/output_ggir_inputs_2025E_{SUFFIX}/results/QC/part5_daysummary_full_MM_L40M100V400_T5A5.csv")
    axivity_activity_df = pd.read_csv(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_axivity_{RUN_VERSION}/output_ggir_inputs_axivity_{SUFFIX}/results/QC/part5_daysummary_full_MM_L40M100V400_T5A5.csv")
    compare_activity_raw(USER, DEVICE, RUN_VERSION, SUFFIX)
    compare_sleep(USER, DEVICE, verisense_sleep_df, axivity_sleep_df)
    compare_activity_summary(USER, DEVICE, verisense_activity_df, axivity_activity_df)
