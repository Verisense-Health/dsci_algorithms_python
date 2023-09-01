import numpy as np
import pandas as pd
import dateutil
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# write a function to read the 4th line of a csv file
def read_line(infile, linei):
    with open(infile, "r") as f:
        for i, line in enumerate(f):
            if(i == linei - 1):
                return line
    return None

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
    n = df.shape[0]
    df["etime"] = np.linspace(start, end, n)
    print(df)
    print(df.describe())
    return df

def parse_axivity(infile, outfile, show_plot):
    df = pd.read_csv(infile)
    df.columns = ["etime", "x", "y", "z"]
    if(show_plot):
        fig, axs = plt.subplots(3, 1, figsize = (15, 10), sharex = True)
        plt.plot(pd.to_datetime(df.etime, unit = "s"), df.x, color = "r")
        plt.plot(pd.to_datetime(df.etime, unit = "s"), df.y, color = "g")
        plt.plot(pd.to_datetime(df.etime, unit = "s"), df.z, color = "b")
        plt.show()
    df["temp"] = 1 * df.shape[0]
    df.to_csv(outfile, index = False)
    return df
    # df = df.rename(columns = {"X": "x", "Y": "y", "Z": "z"})


start = 1692287589.4900
time = np.arange(1692287589.4900, 1692287589.4900 + 3600 * 24 * 5, step = 0.04)
x = np.random.uniform(-8, 8, size = len(time))
y = np.random.uniform(-8, 8, size = len(time))
z = np.random.uniform(-8, 8, size = len(time))
df = pd.DataFrame({"etime": time, "x": x, "y": y, "z": z})
df.to_csv("/Users/lselig/Desktop/lucas_long_fake_test.csv", index = False)

infile = "/Users/lselig/Desktop/test_axivitiy_lucas_0817.csv"
parse_axivity(infile, outfile = infile.replace("lucas", "lucas_parsed"), show_plot = True)
if(False):
    # infile = "/Users/lselig/Desktop/230728_212358_Accel_DEFAULT_CAL_00000.csv"
    infile = "/Users/lselig/Desktop/230731_143255_Accel_DEFAULT_CAL_00607.csv"
    df = parse_imu(infile)

    fig, axs = plt.subplots(4, 1, figsize = (10, 10), sharex = True)
    axs[0].plot(pd.to_datetime(df.etime, unit = "s"), df.x)
    axs[1].plot(pd.to_datetime(df.etime, unit = "s"), df.y)
    axs[2].plot(pd.to_datetime(df.etime, unit = "s"), df.z)
    axs[3].plot(pd.to_datetime(df.etime, unit = "s"), np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2))
    plt.show()




# df = pd.read_csv("/Users/lselig/Desktop/test_gsensor.csv")
df = pd.read_csv("/Users/lselig/Desktop/test_gsensor_1.csv")

# df1 = pd.read_csv("/Users/lselig/Desktop/test_gsensor0.csv")
# df2 = pd.read_csv("/Users/lselig/Desktop/test_gsensor1.csv")
# df3 = pd.read_csv("/Users/lselig/Desktop/test_gsensor2.csv")

# df = pd.concat([df1, df2, df3])
# df = df.sort_values(by = "Date")
# fname =  "All three files combined"
fname = "/Users/lselig/Desktop/test_gsensor1.csv"
# df = pd.read_csv(fname)

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
    for i in range(1, 29):
        samp = [df["X" + str(i)].values[0], df["Y" + str(i)].values[0], df["Z" + str(i)].values[0]]
        samples.append(samp)

start = int(dateutil.parser.parse(df.iloc[0]["Date"]).strftime("%s"))
end = int(dateutil.parser.parse(df.iloc[-1]["Date"]).strftime("%s"))
print(end - start)

df["etime"] = etimes
df = df.sort_values(by = "etime")

window = 10
stride = 3
# start = df.iloc[0].etime + 2*window
# end = df.iloc[-1].etime - 2*window
start = df.iloc[0].etime
end = df.iloc[-1].etime - 2*window
curr = start
# curr = start
# df = pd.DataFrame({"Timestamp": timestamps, "X": [s[0] for s in samples], "Y": [s[1] for s in samples], "Z": [s[2] for s in samples]})
# df = df.sort_values(by = "Timestamp")
# print(curr, end, curr < end)
anchors = []
nsamples = []
while(curr < end):
    slice = df[(df["etime"] >= curr) & (df["etime"] < curr + window)]
    anchors.append(np.nanmedian(slice.etime.values))
    nsamples.append((len(slice) * 28) / window)
    # print(curr, end,  end - curr, len(slice), (len(slice) * 28) / window)
    curr += stride
    # timestamps_plot.append(curr)
    # curr += 1/28

# 25 samples per second
# 5 days * 24 hours * 60 minutes * 60 seconds * 25 samples per second = 10,800,000 samples



plt.title(f"Window: {window} sec\n"
          f"Stride: {stride} sec\n"
          f"Average fs: {np.mean(nsamples): .3f} Hz\n"
          f"nwindows: {len(nsamples)}\n")


plt.scatter(pd.to_datetime(anchors, unit = "s"), nsamples, s = 4, color = "black")
plt.ylabel("fs")
plt.xlabel("Window anchor timestamp")
plt.tight_layout()
plt.show()
# print(timestamps[-1] - timestamps[0])

plt.title(fname)
plt.plot(np.diff(timestamps_plot))
# plt.hist(np.diff(np.where(np.diff(timestamps_plot) == 0)[0]), bins = 10)
# plt.plot(np.where(np.diff(timestamps_plot) == 0)[0])
# plt.xlabel("Difference in samples between duplicated timestamps")
# plt.ylabel("Frequency")
plt.xlabel("Sample")
plt.ylabel("t(i) - t(i-1)")
plt.show()

timestamps = np.array(timestamps)
unique = np.unique(timestamps, return_index = True)[1]
samples = np.array(samples)[unique]
timestamps = timestamps[unique]
fig, axs = plt.subplots(3, 1, figsize = (15, 9), sharex = True)
axs[0].plot(timestamps, [s[0] for s in samples], label = "X", drawstyle = "steps-post")
axs[1].plot(timestamps, [s[1] for s in samples], label = "Y", drawstyle = "steps-post")
axs[2].plot(timestamps, [s[2] for s in samples], label = "Z", drawstyle = "steps-post")
axs[0].set_ylabel("X")
axs[1].set_ylabel("Y")
axs[2].set_ylabel("Z")
fig.suptitle("/Users/lselig/Desktop/test_gsensor.csv")
plt.show()
print(df.head())




