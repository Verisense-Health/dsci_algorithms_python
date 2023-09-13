install.packages("GGIR", dependencies = TRUE)
install.packages('/Users/lselig/Desktop/tmp/GGIR_2.9-0.tar.gz', repos=NULL, type='source')
    
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
      outputdir="/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_axivity_v4",
      studyname = "tmp2",
      rmc.dec=".",
      rmc.firstrow.acc = 1,
      rmc.col.acc = c(2, 3, 4),
      rmc.col.time=1,
      rmc.unit.acc = "g",
      rmc.unit.time = "UNIXsec",
      rmc.sf = 100.0,
      print.filename = TRUE,
      rmc.configtz = "America/Chicago",
      rmc.desiredtz = "America/Chicago",
      do.cal = FALSE)


GGIR(mode=c(1,2,3,4,5),
     datadir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_inputs/ggir_inputs_2025E_rescaled",
     outputdir = "/Users/lselig/Desktop/verisense/codebase/dsci_algorithms_python/data/LS2025E/210202054E02/GGIR/ggir_outputs/ggir_outputs_2025E_v4",
     rmc.dec=".",
     rmc.firstrow.acc = 1,
     rmc.col.acc = c(2, 3, 4),
     rmc.col.time=1,
     rmc.unit.acc = "g",
     rmc.unit.time = "UNIXsec",
     rmc.sf = 32.0,
     print.filename = TRUE,
     do.cal = FALSE,
     rmc.desiredtz = "America/Chicago",
     rmc.configtz = 'America/Chicago',
     rmc.check4timegaps = TRUE,
     threshold.lig = 60, # 40
     threshold.mod = 140, # 100
     threshold.vig = 400 # 400
     )

