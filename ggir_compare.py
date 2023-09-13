import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

verisense_acc = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_rescaled/verisense_acc.csv")
axivity_acc = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity/axivity_acc.csv")

# min_i = -8.0
# max_i = 8.0
# verisense_acc.x = min_i + (((verisense_acc.x - np.min(verisense_acc.x)) * (max_i - min_i)) / (np.max(verisense_acc.x) - np.min(verisense_acc.x)))
# verisense_acc.y = min_i + (((verisense_acc.y - np.min(verisense_acc.y)) * (max_i - min_i)) / (np.max(verisense_acc.y) - np.min(verisense_acc.y)))
# verisense_acc.z = min_i + (((verisense_acc.z - np.min(verisense_acc.z)) * (max_i - min_i)) / (np.max(verisense_acc.z) - np.min(verisense_acc.z)))
# verisense_acc = verisense_acc[verisense_acc.etime <= 1694192167]
# verisense_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_rescaled/verisense_acc.csv", index = False)
# print(verisense_acc.head())
# fig, axs = plt.subplots(3, 1, figsize = (15, 9))
# fig.suptitle("Lucas Longitudinal Data: Verisense vs. Axivity")
# axs[0].plot(verisense_acc.etime, verisense_acc.x, label = "Verisense", alpha = 0.6, color = "black")
# axs[0].plot(axivity_acc.etime, axivity_acc.x, label = "Axivity", alpha = 0.6, color = "red")
#
# axs[1].plot(verisense_acc.etime, verisense_acc.y, label = "Verisense", alpha = 0.6, color = "black")
# axs[1].plot(axivity_acc.etime, axivity_acc.y, label = "Axivity", alpha = 0.6, color = "red")
#
# axs[2].plot(verisense_acc.etime, verisense_acc.z, label = "Verisense", alpha = 0.6, color = "black")
# axs[2].plot(axivity_acc.etime, axivity_acc.z, label = "Axivity", alpha = 0.6, color = "red")
#
# plt.show()
# print(verisense_acc.shape[0] / (verisense_acc.iloc[-1].etime - verisense_acc.iloc[0].etime))
# print(axivity_acc.shape[0] / (axivity_acc.iloc[-1].etime - axivity_acc.iloc[0].etime))
# verisense_acc = verisense_acc[["etime", "x", "y", "z"]]
# axivity_acc = axivity_acc[["etime", "x", "y", "z"]]
# verisense_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_rescaled/verisense_acc.csv", index = False)
# axivity_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity/axivity_acc.csv", index = False)
#
# min_x = np.min(axivity_acc.x)
# min_y = np.min(axivity_acc.y)
# min_z = np.min(axivity_acc.z)
#
# max_x = np.max(axivity_acc.x)
# max_y = np.max(axivity_acc.y)
# max_z = np.max(axivity_acc.z)
#
#
# verisense_acc.mag = np.sqrt(verisense_acc.x ** 2 + verisense_acc.y ** 2 + verisense_acc.z ** 2)
#
# verisense_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_rescaled/verisense_acc.csv", index = False)
# axivity_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity_rescaled/axivity_acc.csv", index = False)


def replace_gaps(df):
    print(df.shape[0] / (df.iloc[-1].etime - df.iloc[0].etime))
    diffs = np.diff(df.etime)
    gaps = np.where(diffs > 1)[0]
    gap_fillers = []
    # fig, axs = plt.subplots(4, 1, sharex = True)
    # axs[0].plot(df.etime.values[:-1] ,diffs)
    # axs[1].plot(df.etime, df.x)
    for gap in gaps:
        start = df.iloc[gap].etime
        end = df.iloc[gap + 1].etime
        ts = np.linspace(start, end, int((end - start) * 31.25), endpoint = False)
        x = np.zeros(ts.shape[0])
        y = np.zeros(ts.shape[0])
        z = np.ones(ts.shape[0])
        gap_fillers.append(pd.DataFrame({"etime": ts, "x": x, "y": y, "z": z}))
    gap_fillers = pd.concat(gap_fillers)
    df = pd.concat([df, gap_fillers])
    df = df.sort_values(by = "etime")
    df = df.drop_duplicates()
    df["etime"] = np.linspace(df.iloc[0].etime, df.iloc[-1].etime, df.shape[0])
    print(df.shape[0] / (df.iloc[-1].etime - df.iloc[0].etime))
    # axs[2].plot(df.etime, df.x)
    # axs[3].plot(df.etime.values[:-1] ,np.diff(df.etime))
    # plt.show()
    return df


# print(replace_gaps(verisense_acc))
# verisense_acc = replace_gaps(verisense_acc)
# print("writing")
# verisense_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_rescaled/verisense_acc.csv", index = False)
# print("wrote")
#
# plt.plot(pd.to_datetime(verisense_acc.etime, unit = "s"), np.sqrt(verisense_acc.x ** 2 + verisense_acc.y ** 2 + verisense_acc.z ** 2), alpha = 0.6, label = "Verisense")
# plt.plot(pd.to_datetime(axivity_acc.etime, unit = "s"), np.sqrt(axivity_acc.x ** 2 + axivity_acc.y ** 2 + axivity_acc.z ** 2), alpha = 0.6, label = "Axivity")
# plt.xlabel("Time")
# plt.ylabel("Magnitude (G)")
# plt.title("Lucas Longitudinal Data: Verisense vs. Axivity")
# plt.legend()
# plt.show()

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


def compare_activity(USER, DEVICE, verisense_activity_df, axivity_activity_df):
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



df = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity/axivity_acc.csv")
print(df.head())
USER = "LS2025E"
DEVICE = "210202054E02"
verisense_sleep_df = pd.read_csv(
    f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_2025E_v2/output_ggir_inputs_2025E/results/part4_nightsummary_sleep_cleaned.csv")
axivity_sleep_df = pd.read_csv(
    f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{USER}/{DEVICE}/GGIR/ggir_outputs/ggir_outputs_axivity_v2/output_ggir_inputs_axivity/results/part4_nightsummary_sleep_cleaned.csv")

trial_name = "rescaled"
verisense_activity_df = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v4/output_ggir_inputs_2025E_rescaled/results/QC/part5_daysummary_full_MM_L60M140V400_T5A5.csv")
axivity_activity_df = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v4/output_ggir_inputs_axivity/results/QC/part5_daysummary_full_MM_L40M100V400_T5A5.csv")
# compare_sleep(USER, DEVICE, verisense_sleep_df, axivity_sleep_df)
compare_activity(USER, DEVICE, verisense_activity_df, axivity_activity_df)
