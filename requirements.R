# requirements.R
# Run this file first to install all required packages
# source("requirements.R")

if (!require("tidyverse"))  install.packages("tidyverse")
if (!require("cluster"))    install.packages("cluster")
if (!require("factoextra")) install.packages("factoextra")

library(tidyverse)
library(cluster)
library(factoextra)

cat("All packages installed and loaded successfully.\n")
