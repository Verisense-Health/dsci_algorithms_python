import numpy as np
import pandas as pd
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
