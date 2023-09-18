import awswrangler as wr
import pandas as pd
import numpy as np
from alive_progress import alive_bar
import seaborn as sns
import boto3, glob
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pytz

sns.set_style("darkgrid")
AWS_ACCESS_KEY = "AKIAR2C2O5V35DS42JAQ"
AWS_SECRET_KEY = "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"
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
        start = np.where(etime == u)[0][0]
        end = np.where(etime == u)[0][-1]
        chunk_ts = np.linspace(u, u + 1, end - start, endpoint = False)
        copy_df["etime"][start:end] = chunk_ts

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
def parse_green_ppg(infile, show_plot = False):
    df = pd.read_csv(infile, skiprows=9)
    df.columns = ["etime", "green"]
    df["etime"] = df["etime"] / 1000
    df = df.sort_values(by="etime")
    if(show_plot):
        plt.plot(df.etime, df.green)
        plt.xlabel("Time (s)")
        plt.ylabel("Green PPG (a.u.)")
        plt.title(infile.split("/")[-1])
        plt.show()
    return df
def parse_red_ppg(infile, show_plot = False):
    # print(infile)
    df = pd.read_csv(infile, skiprows=9)
    df.columns = ["etime", "red"]
    df["etime"] = df["etime"] / 1000
    df = df.sort_values(by="etime")
    if (show_plot):
        plt.plot(df.etime, df.red)
        plt.xlabel("Time (s)")
        plt.ylabel("Red PPG (a.u.)")
        plt.title(infile.split("/")[-1])
        plt.show()
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
def download_signal(bucket,
                    user,
                    device,
                    signal,
                    after):

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY)

    # List objects in the bucket that contain the specified substring
    objects_with_substring = wr.s3.list_objects(path=f"s3://{bucket}/1/{user}/{device}/ParsedFiles/", suffix=f"{signal}.csv",
                                                boto3_session=session,
                                                last_modified_begin = datetime.strptime(after, "%Y-%m-%d").astimezone(pytz.timezone("US/Central")))

    objects_with_substring_bin = wr.s3.list_objects(path=f"s3://{bucket}/1/{user}/{device}/BinaryFiles/", suffix=f"{signal}.bin2",
                                                boto3_session=session,
                                                last_modified_begin = datetime.strptime(after, "%Y-%m-%d").astimezone(pytz.timezone("US/Central")))

    missing_some = False
    assert(len(objects_with_substring) == len(objects_with_substring_bin))
    for obj in objects_with_substring_bin:
        obj_name = obj.split(".")[0].split("/")[-1]
        have = False
        for obj2 in objects_with_substring:
            if(obj_name in obj2):
                have = True
                break
        if(not have):
            print("MISSING PARSED: ", obj)
            missing_some = True

    if(missing_some):
        return -1
        a = 1


    for obj in objects_with_substring:
        print("Object Key:", obj)
        saveloc = Path(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{user}/{device}/{signal}")
        if(not saveloc.exists()):
            Path(saveloc).mkdir(parents=True, exist_ok=True)
        if(not Path(f"{str(saveloc)}/{obj.split('/')[-1]}").exists()):
            wr.s3.download(path=obj, local_file=f"{str(saveloc)}/{obj.split('/')[-1]}", boto3_session=session)
            print(f"Downloaded: {obj}")
        if("230901_181617_RedPPG.csv" in obj):
            wr.s3.download(path=obj, local_file=f"{str(saveloc)}/{obj.split('/')[-1]}", boto3_session=session)
            print(f"Downloaded: {obj}")

        else:
            print(f"Already downloaded: {obj}")

def parse_verisense_direct_hr(f):
    df = pd.read_csv(f)
    return df
def combine_signal(user, device, signal, outfile, use_cache, after):
    after = datetime.strptime(after, "%Y-%m-%d")
    if(use_cache):
        return pd.read_csv(outfile)
    # files = Path(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{user}/{device}/{signal}").glob("*.csv")
    files = glob.glob(f"/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/{user}/{device}/{signal}/*.csv")
    keep_dfs = []
    print(f"Combining {user}_{device}_{signal}")
    files_subset = []
    for f in files:
        file_date = datetime.strptime(f.split("/")[-1].split("_")[0], "%y%m%d")
        if(file_date < after):
            print("Skipping", f, "because it is before", after)
            continue
        else:
            files_subset.append(f)
    with alive_bar(len(list(files_subset)), force_tty = True) as bar:
        for f in files_subset:
            if("HeartRate" not in signal):
                keep_dfs.append(parse_2025e(f, signal))
            else:
                keep_dfs.append(parse_verisense_direct_hr(f))
            bar()
    df = pd.concat(keep_dfs)
    df = df.sort_values(by = "etime")
    df = df.drop_duplicates()
    df.to_csv(outfile, index = False)
    return df