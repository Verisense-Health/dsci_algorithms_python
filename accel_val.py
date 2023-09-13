from download_2025E_data import download_signal, combine_signal, parse_accel
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parse_third_party import parse_axivity, read_line, parse_imu, parse_shimmer3_accel
from dateutil import parser
sns.set_style("darkgrid")


BUCKET = "verisense-cd1f868f-eada-44ac-b708-3b83f2aaed73"
USER = "LS2025E"
# USER = "GG2025E"
DEVICE = "210202054E02"
# DEVICE = "210202054DFB"
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


def combine_axivity(infolder, outfolder):
    dfs = []
    for file in glob.glob(infolder + "/*.csv"):
        print(f"Parsing {file}")
        df = parse_axivity(file)
        dfs.append(df)

    print(f"Concatenating {len(dfs)} axivity files")
    df = pd.concat(dfs)
    df = df.sort_values(by = "etime")
    df = df.drop_duplicates()
    df.to_csv(f"{outfolder}/axivity_acc.csv", index = False)
    return df


def prep_ggir(axivity_acc_path, verisense_acc_path, start, end):
    axivity_acc = pd.read_csv(axivity_acc_path)
    verisense_acc = pd.read_csv(verisense_acc_path)

    axivity_acc = axivity_acc[axivity_acc.etime.between(start, end)]
    verisense_acc = verisense_acc[verisense_acc.etime.between(start, end)]
    verisense_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E/verisense_acc.csv", index = False)
    axivity_acc.to_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity/axivity_acc.csv", index = False)

    plt.plot(pd.to_datetime(axivity_acc.etime, unit = "s"), axivity_acc.mag, color="C0")
    plt.plot(pd.to_datetime(verisense_acc.etime, unit = "s"), verisense_acc.mag, color = "C1")
    plt.show()

def compare_imu_2025():
    muaaz_2025_1 = parse_accel(
        "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_accel_range_test/2025e watch/230911_164442_Accel_X.csv")
    muaaz_2025_2 = parse_accel(
        "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_accel_range_test/2025e watch/230911_164715_Accel_Y.csv")
    muaaz_2025_3 = parse_accel(
        "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_accel_range_test/2025e watch/230911_165027_Accel_Z.csv")
    muaaz_2025 = pd.concat([muaaz_2025_1, muaaz_2025_2, muaaz_2025_3])

    muaaz_imu_1 = parse_imu(
        "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_accel_range_test/Verisense device/230912_100046_Accel_DEFAULT_X.csv",
        custom_start=muaaz_2025_1.iloc[0].etime, custom_end=muaaz_2025_1.iloc[-1].etime)
    muaaz_imu_2 = parse_imu(
        "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_accel_range_test/Verisense device/230912_100326_Accel_DEFAULT_Y.csv",
        custom_start=muaaz_2025_2.iloc[0].etime, custom_end=muaaz_2025_2.iloc[-1].etime)
    muaaz_imu_3 = parse_imu(
        "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/muaaz_accel_range_test/Verisense device/230912_100840_Accel_DEFAULT_Z.csv",
        custom_start=muaaz_2025_3.iloc[0].etime, custom_end=muaaz_2025_3.iloc[-1].etime)
    muaaz_imu = pd.concat([muaaz_imu_1, muaaz_imu_2, muaaz_imu_3])

    fig, axs = plt.subplots(4, 1, figsize=(15, 9))
    axs[0].plot(pd.to_datetime(muaaz_2025.etime, unit="s"), muaaz_2025.x, label="2025E", alpha=0.6, color="black")
    axs[1].plot(pd.to_datetime(muaaz_2025.etime, unit="s"), muaaz_2025.y, label="2025E", alpha=0.6, color="black")
    axs[2].plot(pd.to_datetime(muaaz_2025.etime, unit="s"), muaaz_2025.z, label="2025E", alpha=0.6, color="black")
    axs[3].plot(pd.to_datetime(muaaz_2025.etime, unit="s"), muaaz_2025.mag, label="2025E", alpha=0.6, color="black")

    axs[0].plot(pd.to_datetime(muaaz_imu.etime, unit="s"), muaaz_imu.x, label="Verisense IMU", alpha=0.6, color="red")
    axs[1].plot(pd.to_datetime(muaaz_imu.etime, unit="s"), muaaz_imu.y, label="Verisense IMU", alpha=0.6, color="red")
    axs[2].plot(pd.to_datetime(muaaz_imu.etime, unit="s"), muaaz_imu.z, label="Verisense IMU", alpha=0.6, color="red")
    axs[3].plot(pd.to_datetime(muaaz_imu.etime, unit="s"), muaaz_imu.mag, label="Verisense IMU", alpha=0.6, color="red")
    for i in range(4):
        axs[i].legend()

    axs[0].set_ylabel("X (g)")
    axs[1].set_ylabel("Y (g)")
    axs[2].set_ylabel("Z (g)")
    axs[3].set_ylabel("Mag (g)")

    plt.show()


def compare_shimmer3_2025():
    watch1 = parse_accel("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_accel_range_test/accel range test 1 x axis watch data.csv")
    watch2 = parse_accel("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_accel_range_test/accel range test 1 y axis watch data.csv")
    watch3 = parse_accel("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_accel_range_test/accel range test 1 z axis watch data.csv")
    watch = pd.concat([watch1, watch2, watch3])

    shimmer1 = parse_shimmer3_accel("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_accel_range_test/accel range test 1 x axis shimmer data.csv")
    shimmer2 = parse_shimmer3_accel("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_accel_range_test/accel range test 1 y axis shimmer data.csv")
    shimmer3 = parse_shimmer3_accel("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/ww_accel_range_test/accel range test 1 z axis shimmer data.csv")
    shimmer = pd.concat([shimmer1, shimmer2, shimmer3])

    fig, axs = plt.subplots(4, 1, figsize=(15, 9), sharex = True)
    axs[0].plot(pd.to_datetime(watch.etime, unit="s"), watch.x, label="2025E", alpha=0.6, color="black")
    axs[1].plot(pd.to_datetime(watch.etime, unit="s"), watch.y, label="2025E", alpha=0.6, color="black")
    axs[2].plot(pd.to_datetime(watch.etime, unit="s"), watch.z, label="2025E", alpha=0.6, color="black")
    axs[3].plot(pd.to_datetime(watch.etime, unit="s"), watch.mag, label="2025E", alpha=0.6, color="black")

    axs[0].plot(pd.to_datetime(shimmer.etime, unit="s"), shimmer.x, label="Shimmer3", alpha=0.6, color="red")
    axs[1].plot(pd.to_datetime(shimmer.etime, unit="s"), shimmer.z, label="Shimmer3", alpha=0.6, color="red")
    axs[2].plot(pd.to_datetime(shimmer.etime, unit="s"), shimmer.y, label="Shimmer3", alpha=0.6, color="red")
    axs[3].plot(pd.to_datetime(shimmer.etime, unit="s"), shimmer.mag, label="Shimmer3", alpha=0.6, color="red")
    for i in range(4):
        axs[i].legend()

    axs[0].set_ylabel("X (g)")
    axs[1].set_ylabel("Y (g)")
    axs[2].set_ylabel("Z (g)")
    axs[3].set_ylabel("Mag (g)")
    print(np.nanmean(shimmer.mag), np.nanmean(watch.mag))

    plt.show()

    fig, axs = plt.subplots(1, 4, figsize=(15, 9))
    axs[0].hist(watch.x, bins = 100, alpha = 0.6, color = "black", label = f"Mean: {np.nanmean(watch.x):.2f} 2025E", density = True)
    axs[0].hist(shimmer.x, bins = 100, alpha = 0.6, color = "red", label = f"Mean: {np.nanmean(shimmer.x):.2f} Shimmer 3", density = True)

    axs[1].hist(watch.y, bins = 100, alpha = 0.6, color = "black", label = f"Mean: {np.nanmean(watch.y):.2f} 2025E", density = True)
    axs[1].hist(shimmer.y, bins = 100, alpha = 0.6, color = "red", label = f"Mean: {np.nanmean(shimmer.y):.2f} Shimmer 3", density = True)

    axs[2].hist(watch.z, bins = 100, alpha = 0.6, color = "black", label = f"Mean: {np.nanmean(watch.z):.2f} 2025E", density = True)
    axs[2].hist(shimmer.z, bins = 100, alpha = 0.6, color = "red", label = f"Mean: {np.nanmean(shimmer.z):.2f} Shimmer 3", density = True)

    axs[3].hist(watch.mag, bins = 100, alpha = 0.6, color = "black", label = f"Mean: {np.nanmean(watch.mag):.2f} 2025E", density = True)
    axs[3].hist(shimmer.mag, bins = 100, alpha = 0.6, color = "red", label = f"Mean: {np.nanmean(shimmer.mag):.2f} Shimmer 3", density = True)
    axs[0].set_xlabel("X")
    axs[1].set_xlabel("Y")
    axs[2].set_xlabel("Z")
    axs[3].set_xlabel("Mag")
    for i in range(4):
        axs[i].legend()
        axs[i].set_ylabel("Counts")
    plt.show()





def main():

    # compare_shimmer3_2025()
    # compare_imu_2025()


    # df = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/verisense_acc.csv")
    # print(df.head(300), df.shape)

    # df = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity/axivity_acc.csv")
    # print(df.head(300), df.shape)
    # plt.plot(df.etime, df.mag)
    # plt.plot(df2.etime, df2.mag)
    # plt.show()

    # axivity_in_folder = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/axivity"
    axivity_in_folder = "/Users/lselig/Library/CloudStorage/GoogleDrive-lucas.a.selig@gmail.com/My Drive/verisense/axivity"
    combined_axivity_out_folder = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity"
    # for signal in SIGNALS:
    #     download_signal(BUCKET, USER, DEVICE, signal, after = "2023-09-01")
    #
    combine_axivity(axivity_in_folder, combined_axivity_out_folder)
    axivity_acc = pd.read_csv(f"{combined_axivity_out_folder}/axivity_acc.csv")
    verisense_acc = combine_signal(USER, DEVICE, signal ="Accel", outfile = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E/verisense_acc.csv", use_cache = False, after = "2023-09-01")

    # axivity_acc = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity/axivity_acc.csv")
    # verisense_acc = pd.read_csv("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E/verisense_acc.csv")
    print(axivity_acc.head())
    print(axivity_acc.shape[0] / (axivity_acc.iloc[-1].etime - axivity_acc.iloc[0].etime))
    print(verisense_acc.shape[0] / (verisense_acc.iloc[-1].etime - verisense_acc.iloc[0].etime))

    print(verisense_acc.head())
    plt.plot(verisense_acc.etime, verisense_acc.mag, alpha = 0.6)
    plt.plot(axivity_acc.etime, axivity_acc.mag, alpha = 0.6)
    plt.show()
    # axivity_acc =
    prep_ggir("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity/axivity_acc.csv",
              "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E/verisense_acc.csv",
              start = 1693956574,
              end = np.inf)

    # axivity_acc = parse_axivity("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/trials/acc_range_test/0901_day_axivity.csv")
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
