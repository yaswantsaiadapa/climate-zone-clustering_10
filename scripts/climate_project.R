̥̥#install req packages
install.packages("tidyverse")
install.packages("cluster")
install.packages("factoextra")
#Libraries for data manipulation and clustering
library(tidyverse)
library(cluster)
library(factoextra)
#Loading data set
df <- read.csv("C:\\Users\\yaswa\\OneDrive\\Desktop\\projects\\climate-zone-clustering\\GlobalWeatherRepository.csv")

head(df)
#Inspect data set
#checking structure and missing values
dim(df)
colnames(df)
str(df)
colSums(is.na(df))

#Select relevant features(only numericals)
climate_df <- df %>%
  select(
    temperature_celsius,
    humidity,
    wind_kph,
    pressure_mb,
    precip_mm,
    cloud,
    visibility_km,
    uv_index,
    air_quality_PM2.5,
    air_quality_PM10
  )

head(climate_df)

#Handle with missing vals(replace NA with col mean)
#Using Z score standardizn
for(i in 1:ncol(climate_df)){
  climate_df[is.na(climate_df[,i]), i] <- mean(climate_df[,i], na.rm = TRUE)
}

colSums(is.na(climate_df))

#Scale data(standardizing)
scaled_data <- scale(climate_df)

head(scaled_data)

#-----PART-2------
#Elbow method(got k=4)
set.seed(123)

wss <- numeric(10)

for (k in 1:10) {
  km <- kmeans(scaled_data, centers = k, nstart = 10)
  wss[k] <- km$tot.withinss
}

wss
plot(
  1:10, 
  wss, 
  type = "b",
  pch = 19,
  frame = FALSE,
  xlab = "Number of Clusters (K)",
  ylab = "Total Within-Cluster Sum of Squares",
  main = "Elbow Method"
)
#Run final clustring model
set.seed(123)
kmeans_result <- kmeans(scaled_data, centers = 4, nstart = 25)
#Attatch cluster numbers for each city
df$Cluster <- as.factor(kmeans_result$cluster)

head(df[, c("location_name", "Cluster")])

kmeans_result

#Understands Cluster characteristics and Interpret with the Output
aggregate(climate_df, by = list(Cluster = df$Cluster), mean)

#----Climate Zone Labeling----
# Based on aggregate() means:
# Cluster 1: Hot(26.5C), very dry(35.5%), near-zero rain, extreme PM2.5(231) -> High Pollution Zone
#            PM2.5 is 9x other clusters — pollution is the unifying signal, not aridity
# Cluster 2: Cool(16.4C), moderate humidity(69%), low UV(1.3), clean air(25 PM2.5) -> Cool Temperate
#            All sample cities confirmed temperate (Buenos Aires, Canberra, Baku etc.)
# Cluster 3: Warm(19.5C), highest humidity(83%), heavy cloud(74%), most rain(0.31mm) -> Humid & Overcast
#            Captures high humidity globally — includes European cities, not just tropics
# Cluster 4: Hottest(28.6C), highest UV(7.5), dry(44%), clear skies -> Hot & Sunny
#            Algiers, Luanda, Manama confirm hot arid/semi-arid zones
df$ClimateZone <- dplyr::recode(as.character(df$Cluster),
                                "1" = "High Pollution Zone",
                                "2" = "Cool Temperate",
                                "3" = "Humid & Overcast",
                                "4" = "Hot & Sunny"
)

cat("Climate Zone Distribution:\n")
print(table(df$ClimateZone))

cat("\nSample cities per climate zone:\n")
for(zone in c("High Pollution Zone", "Cool Temperate", "Humid & Overcast", "Hot & Sunny")) {
  cities <- df$location_name[df$ClimateZone == zone]
  cat(zone, "->", paste(head(unique(cities), 5), collapse = ", "), "\n")
}

#------PART-3-------
#Extreme weather Outlier detection
#Using Z score
#Detect temperature Outliers(extreme heat and cold temp)
# Note: abs(z) > 3 here catches general temperature outliers both directions.
# Separate heatwave/coldwave counts using percentile method are in Part 4.
temp_z <- as.vector(scale(climate_df$temperature_celsius))

heatwave_index <- which(abs(temp_z) > 3)

heatwave_cities <- df[heatwave_index, c("location_name", "temperature_celsius")]

heatwave_cities
#Detect Heavy rain fall outliers
rain_z <- scale(climate_df$precip_mm)

heavy_rain_index <- which(abs(rain_z) > 3)

heavy_rain_cities <- df[heavy_rain_index, c("location_name", "precip_mm")]

heavy_rain_cities
#Detect strong wind outliers
wind_z <- scale(climate_df$wind_kph)

strong_wind_index <- which(abs(wind_z) > 3)

strong_wind_cities <- df[strong_wind_index, c("location_name", "wind_kph")]

strong_wind_cities
#Detect dangerous pollution
pm25_z <- scale(climate_df$air_quality_PM2.5)

pollution_index <- which(abs(pm25_z) > 3)

polluted_cities <- df[pollution_index, c("location_name", "air_quality_PM2.5")]

polluted_cities
#Count total outliers to get an idea
length(heatwave_index)
length(heavy_rain_index)
length(strong_wind_index)
length(pollution_index)


#---PART-4-----
#Visualizing K means cluster(diff colors diff clusters)
#interpret form the op image
fviz_cluster(
  kmeans_result,
  data = scaled_data,
  ellipse.type = "convex",
  palette = "jco",
  ggtheme = theme_minimal(),
  main = "Climate Zone Clustering"
)

#Temperature distribution plot
boxplot(
  climate_df$temperature_celsius,
  main = "Temperature Distribution",
  col = "lightblue",
  ylab = "Temperature (°C)"
)

#Clean Outlier summary

#Temperature Extremes
# Z-score threshold of 3 is too strict for this dataset:
# mean~22C, SD~13C means heatwave threshold would be ~58C which no city reaches.
# Percentile-based detection is more appropriate for skewed global temperature data.
heat_threshold <- quantile(climate_df$temperature_celsius, 0.95)
cold_threshold <- quantile(climate_df$temperature_celsius, 0.05)

heatwave_index <- which(climate_df$temperature_celsius > heat_threshold)
coldwave_index <- which(climate_df$temperature_celsius < cold_threshold)

cat("Heatwave threshold (95th percentile):", round(heat_threshold, 2), "C\n")
cat("Coldwave threshold (5th percentile): ", round(cold_threshold, 2), "C\n")
cat("Number of Heatwaves:", length(heatwave_index), "\n")
cat("Number of Coldwaves:", length(coldwave_index), "\n")

#Show heatwave cities
heatwave_cities <- df[heatwave_index, c("location_name", "temperature_celsius")]
cat("Heatwave Cities (sample):\n")
print(head(heatwave_cities, 10))

#Show coldwave cities
coldwave_cities <- df[coldwave_index, c("location_name", "temperature_celsius")]
cat("Coldwave Cities (sample):\n")
print(head(coldwave_cities, 10))

#Heavy Rainfall
rain_z <- scale(climate_df$precip_mm)
heavy_rain_index <- which(rain_z > 3)
cat("Heavy Rainfall Events:", length(heavy_rain_index), "\n")

#Strong wind events
wind_z <- scale(climate_df$wind_kph)
strong_wind_index <- which(wind_z > 3)
cat("Strong Wind Events:", length(strong_wind_index), "\n")

#High pollution
pm25_z <- scale(climate_df$air_quality_PM2.5)
pollution_index <- which(pm25_z > 3)
cat("Dangerous Pollution Events:", length(pollution_index), "\n")


#----Hierarchial Clustering---
#Take sample data instead of getting crashes
set.seed(123)
sample_index <- sample(1:nrow(scaled_data), 2000)
sample_data <- scaled_data[sample_index, ]
#compute distance on sample
dist_matrix <- dist(sample_data)
#Hierarchial clustering
hc_result <- hclust(dist_matrix, method = "ward.D2")
#Plot dendogram
plot(hc_result, labels = FALSE,
     main = "Hierarchical Clustering (Sampled Data)")


#----PART 5: Seasonal Trend Analysis----
# Parse the last_updated timestamp column into date components
# GlobalWeatherRepository uses format: "2024-01-01 06:00"
df$last_updated <- as.POSIXct(df$last_updated, format = "%Y-%m-%d %H:%M")

# Verify parsing worked — should show POSIXct, not NA
cat("Timestamp sample:\n")
print(head(df$last_updated))

# Extract month and assign meteorological season
df$month <- as.integer(format(df$last_updated, "%m"))

df$season <- case_when(
  df$month %in% c(12, 1, 2) ~ "Winter",
  df$month %in% c(3, 4, 5)  ~ "Spring",
  df$month %in% c(6, 7, 8)  ~ "Summer",
  df$month %in% c(9, 10, 11) ~ "Autumn"
)

# Confirm season distribution — check no NAs
cat("\nRecords per season:\n")
print(table(df$season))

# Attach climate_df columns back for seasonal aggregation
df$temperature_celsius <- climate_df$temperature_celsius
df$humidity            <- climate_df$humidity
df$precip_mm           <- climate_df$precip_mm
df$wind_kph            <- climate_df$wind_kph
df$air_quality_PM2.5   <- climate_df$air_quality_PM2.5

# Seasonal summary — mean of key variables per season
seasonal_summary <- df %>%
  group_by(season) %>%
  summarise(
    avg_temp      = round(mean(temperature_celsius, na.rm = TRUE), 2),
    avg_humidity  = round(mean(humidity,            na.rm = TRUE), 2),
    avg_precip    = round(mean(precip_mm,           na.rm = TRUE), 2),
    avg_wind      = round(mean(wind_kph,            na.rm = TRUE), 2),
    avg_pollution = round(mean(air_quality_PM2.5,   na.rm = TRUE), 2),
    total_records = n()
  ) %>%
  arrange(factor(season, levels = c("Spring", "Summer", "Autumn", "Winter")))

cat("\nSeasonal Summary Table:\n")
print(seasonal_summary)

# Monthly average temperature trend (to see intra-year pattern)
monthly_temp <- df %>%
  group_by(month) %>%
  summarise(avg_temp = round(mean(temperature_celsius, na.rm = TRUE), 2)) %>%
  arrange(month)

cat("\nMonthly Average Temperature:\n")
print(monthly_temp)

#----Seasonal Visualizations----

# Plot 1: Seasonal avg temperature bar chart
season_order <- c("Spring", "Summer", "Autumn", "Winter")
seasonal_summary$season <- factor(seasonal_summary$season, levels = season_order)

barplot(
  seasonal_summary$avg_temp,
  names.arg = seasonal_summary$season,
  col       = c("#8BC34A", "#FF7043", "#FF8F00", "#42A5F5"),
  main      = "Average Temperature by Season",
  ylab      = "Temperature (°C)",
  xlab      = "Season",
  ylim      = c(0, max(seasonal_summary$avg_temp) * 1.2)
)

# Plot 2: Monthly temperature trend line
plot(
  monthly_temp$month,
  monthly_temp$avg_temp,
  type  = "b",
  pch   = 19,
  col   = "#E53935",
  xaxt  = "n",
  main  = "Monthly Average Temperature Trend",
  xlab  = "Month",
  ylab  = "Avg Temperature (°C)",
  frame = FALSE
)
axis(1,
     at     = 1:12,
     labels = c("Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec")
)

# Plot 3: Seasonal boxplot — full temperature distribution per season
boxplot(
  temperature_celsius ~ season,
  data  = df[df$season %in% season_order, ],
  col   = c("#FFB74D", "#FF7043", "#8BC34A", "#42A5F5"),
  main  = "Temperature Distribution by Season",
  xlab  = "Season",
  ylab  = "Temperature (°C)",
  names = season_order
)

# Plot 4: Seasonal precipitation bar chart
barplot(
  seasonal_summary$avg_precip,
  names.arg = seasonal_summary$season,
  col       = c("#64B5F6", "#1565C0", "#0288D1", "#B3E5FC"),
  main      = "Average Precipitation by Season",
  ylab      = "Precipitation (mm)",
  xlab      = "Season"
)

# Seasonal anomaly count — heatwaves and coldwaves per season
df$is_heatwave <- climate_df$temperature_celsius > heat_threshold
df$is_coldwave <- climate_df$temperature_celsius < cold_threshold

seasonal_anomalies <- df %>%
  group_by(season) %>%
  summarise(
    heatwaves = sum(is_heatwave, na.rm = TRUE),
    coldwaves = sum(is_coldwave, na.rm = TRUE)
  ) %>%
  arrange(factor(season, levels = season_order))

cat("\nSeasonal Anomaly Counts:\n")
print(seasonal_anomalies)


# Save seasonal anomaly chart
png("seasonal_anomaly_chart.png", width=800, height=500)
barplot(
  rbind(c(1319, 3790, 848, 264), c(296, 81, 896, 5044)),
  beside = TRUE,
  names.arg = c("Spring", "Summer", "Autumn", "Winter"),
  col = c("#EF9F27", "#85B7EB"),
  legend.text = c("Heatwaves", "Coldwaves"),
  main = "Seasonal Anomaly Counts — Heatwaves vs Coldwaves",
  ylab = "Number of Events",
  xlab = "Season"
)
dev.off()
