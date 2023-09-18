import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def my_minmax(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


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

def replace_gaps(df, show_plot = True):
    print(df.shape[0] / (df.iloc[-1].etime - df.iloc[0].etime))
    diffs = np.diff(df.etime)
    gaps = np.where(diffs > 0.5)[0]
    gap_fillers = []
    if(show_plot):
        fig, axs = plt.subplots(4, 1, sharex = True)
        axs[0].plot(pd.to_datetime(df.etime.values[:-1] , unit = "s"),diffs)
        axs[1].plot(pd.to_datetime(df.etime, unit = "s"), df.x)

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

    if(show_plot):
        axs[2].plot(pd.to_datetime(df.etime, unit = "s"), df.x)
        axs[3].plot(pd.to_datetime(df.etime.values[:-1] , unit = "s"),np.diff(df.etime))
        plt.show()
    return df

