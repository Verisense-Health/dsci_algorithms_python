install.packages("GGIR", dependencies = TRUE)
library(GGIR)


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

GGIR(mode=c(1,2,3,4,5),
      datadir="/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_axivity",
      outputdir="/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity",
      studyname = "tmp",
      rmc.dec=".",
      rmc.firstrow.acc = 1,
      rmc.firstrow.header=c(),
      rmc.col.acc = c(2, 3, 4),
      rmc.col.time=1,
      rmc.unit.acc = "g",
      rmc.unit.temp = "C",
      rmc.unit.time = "UNIXsec",
      rmc.sf = 100,
      print.filename = TRUE)

GGIR(mode=c(1,2,3,4,5),
     datadir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E",
     outputdir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E",
     #datadir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/GG2025E/210202054DFB/ggir_inputs",
     #outputdir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/GG2025E/210202054DFB/ggir_outputs",     
     rmc.dec=".",
     rmc.firstrow.acc = 2,
     rmc.col.acc = c(1, 2, 3),
     rmc.col.time=5,
     rmc.unit.acc = "g",
     rmc.unit.time = "UNIXsec",
     rmc.sf = 31.25,
     print.filename = TRUE,
     do.cal = FALSE,
     )

