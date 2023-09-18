install.packages("GGIR", dependencies = TRUE)

# Libraries
library(GGIR)
library(ggplot2)
library(dplyr)

load("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v5/output_ggir_inputs_2025E_clean/meta/basic/meta_verisense_acc.csv.RData")
etime <- M$metashort$timestamp
ENMO <- M$metashort$ENMO
data <- data.frame(etime = etime,
                   mag = ENMO)
write.csv(data, "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v5/output_ggir_inputs_2025E/meta/basic/verisense_ggir_metrics.csv", row.names=FALSE)

load("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v5/output_ggir_inputs_axivity_clean/meta/basic/meta_axivity_acc.csv.RData")
etime <- M$metashort$timestamp
ENMO <- M$metashort$ENMO
data <- data.frame(etime = etime,
                   mag = ENMO)
write.csv(data, "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v5/output_ggir_inputs_axivity/meta/basic/axivity_ggir_metrics.csv", row.names=FALSE)


# verisense
load("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v5/output_ggir_inputs_2025E_clean/meta/ms2.out/verisense_acc.csv.RData")
etime <- IMP$metashort$timestamp
ENMO <- IMP$metashort$ENMO
anglez <- IMP$metashort$anglez
data <- data.frame(etime = etime,
                   mag = ENMO,
                   anglez = anglez)
write.csv(data, "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v5/output_ggir_inputs_2025E_clean/meta/ms2.out/verisense_acc.csv", row.names=FALSE)

output$nonwear_perc_day
# axivity
load("/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v5/output_ggir_inputs_axivity_clean/meta/ms2.out/axivity_acc.csv.RData")
etime <- IMP$metashort$timestamp
ENMO <- IMP$metashort$ENMO
anglez <- IMP$metashort$anglez
data <- data.frame(etime = etime,
                   mag = ENMO,
                   anglez = anglez)
write.csv(data, "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v5/output_ggir_inputs_axivity_clean/meta/ms2.out/axivity_acc.csv", row.names=FALSE)



p <- ggplot(data, aes(x=etime, y=mag)) +
    geom_line() + ''
    xlab("")
p

loadedData = read.myacc.csv(rmc.file="/Users/lselig/Documents/joint_corp/watch_accel.csv",
                            rmc.dec=".",
                            rmc.firstrow.acc = 1,
                            rmc.firstrow.header=c(),
                            rmc.col.acc = c(1, 2, 3),
                            rmc.col.time=5,
                            rmc.unit.acc = "g",
                            rmc.unit.time = "UNIXsec")

head(loadedData)
if (file.exists(testfile)) file.remove(testfile)

# list.files("/Users/lselig/Documents/axivity")[1]

C_verisense <- g.calibrate(datafile = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_clean/verisense_acc.csv",
            rmc.dec=".",
            rmc.firstrow.acc = 1,
            rmc.col.acc = c(2, 3, 4),
            rmc.col.time=1,
            rmc.unit.acc = "g",
            rmc.unit.time = "UNIXsec",
            rmc.sf = 32.0,
)

C_axivity <- g.calibrate(datafile = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity_clean/axivity_acc.csv",
                           rmc.dec=".",
                           rmc.firstrow.acc = 1,
                           rmc.col.acc = c(2, 3, 4),
                           rmc.col.time=1,
                           rmc.unit.acc = "g",
                           rmc.unit.time = "UNIXsec",
                           rmc.sf = 100.0,
)

GGIR(mode=c(1,2,3,4,5),
      datadir="/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity_clean",
      outputdir="/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v5",
      studyname = "tmp2",
      rmc.dec=".",
      rmc.firstrow.acc = 1,
      rmc.col.acc = c(2, 3, 4),
      rmc.col.time=1,
      rmc.unit.acc = "g",
      rmc.unit.time = "UNIXsec",
      rmc.sf = 100.0,
     frag.metrics="all",
     part5_agg2_60seconds=TRUE,
      print.filename = TRUE,
      rmc.configtz = "America/Chicago",
      rmc.desiredtz = "America/Chicago",
     save_ms5rawlevels = TRUE,
     epochvalues2csv = TRUE,
     #scale = c(C_verisense$scale[1], C_verisense$scale[2], C_verisense$scale[3]),
     #offset = c(C_verisense$offset[1], C_verisense$offset[2], C_verisense$offset[3]),
     minloadcrit = 24,
     printsummary = TRUE,
     overwrite = TRUE)

GGIR(mode=c(1,2,3,4,5),
     datadir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_clean",
     outputdir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v5",
     rmc.dec=".",
     rmc.firstrow.acc = 1,
     rmc.col.acc = c(2, 3, 4),
     rmc.col.time=1,
     rmc.unit.acc = "g",
     rmc.unit.time = "UNIXsec",
     rmc.sf = 32.0,
     frag.metrics="all",
     part5_agg2_60seconds=TRUE,
     print.filename = TRUE,
     do.cal = TRUE,
     rmc.desiredtz = "America/Chicago",
     rmc.configtz = 'America/Chicago',
     save_ms5rawlevels = TRUE,
     epochvalues2csv = TRUE,
     #scale = c(C_verisense$scale[1], C_verisense$scale[2], C_verisense$scale[3]),
     #offset = c(C_verisense$offset[1], C_verisense$offset[2], C_verisense$offset[3]),
     minloadcrit = 24,
     printsummary = TRUE,
     overwrite = TRUE)
     #threshold.lig = 60, # 40
     #threshold.mod = 140, # 100
     #threshold.vig = 400 # 400


GGIR(mode=c(1,2,3,4,5),
     datadir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_longitudinal",
     outputdir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_longitudinal",
     rmc.dec=".",
     rmc.firstrow.acc = 1,
     rmc.col.acc = c(2, 3, 4),
     rmc.col.time=1,
     rmc.unit.acc = "g",
     rmc.unit.time = "UNIXsec",
     rmc.sf = 32.0,
     frag.metrics="all",
     part5_agg2_60seconds=TRUE,
     print.filename = TRUE,
     do.cal = TRUE,
     rmc.desiredtz = "America/Chicago",
     rmc.configtz = 'America/Chicago',
     save_ms5rawlevels = TRUE,
     epochvalues2csv = TRUE,
     #scale = c(C_verisense$scale[1], C_verisense$scale[2], C_verisense$scale[3]),
     #offset = c(C_verisense$offset[1], C_verisense$offset[2], C_verisense$offset[3]),
     minloadcrit = 24,
     printsummary = TRUE,
     overwrite = TRUE)

