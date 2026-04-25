library(ForestTools)
library(terra)
library(sf)
library(parallel)


#=========================================================
# set working space and create folders
#=========================================================

# Define the desired working directory path
working_dir <- '../run/7_watershed'

# Check if the directory exists, if not, create it
if (!dir.exists(working_dir)) {
  dir.create(working_dir, recursive = TRUE)
}

# Set the working directory to the specified path
setwd(working_dir)
cat("Working directory is ->: ", getwd(), "\n")

# Create folders in the working directory
root <- getwd()
folder_names <- c("6_ttop", "7_crown")

# Loop through each folder name and create the folder
for (name in folder_names) {
  dir.create(file.path(root, name))
}


#=========================================================
# load chm
#=========================================================
# Load sample canopy height model

chm <- terra::rast("../run/5_chm_normalization_tin/chm_tin_tin_mosaic_corrected.tif")
#chm <- terra::rast("./5_chm/chm_01_update_SanRafealSouth.tif")
# # 2D plot
# plot(chm) 

#==================================================
# Detecting treetops by CHM
#==================================================

# Function for defining dynamic window size
# detreetop <- function(x){x * 0.05 + 0.6}
# detreetop <- function(x) { x * 0.07 + 1}
# detreetop <- function(x) { x * 0.02 + 1}
# detreetop <- function(x) { x * 0.02 + 0.5}
detreetop <- function(x){x * 0.02 + 2}

# Detect treetops
ttops <- vwf(chm, winFun = detreetop, minHeight = 2.0)

# Create an sf data frame
sf_ttops <- st_as_sf(ttops, crs = ttops$geomerey)
# Transform sf_ttops to EPSG:4326
sf_ttops_4326 <- st_transform(sf_ttops, crs = 4326)

# Extracting coordinates from geometry column
sf_ttops_4326$lon <- st_coordinates(sf_ttops_4326)[, 1]
sf_ttops_4326$lat <- st_coordinates(sf_ttops_4326)[, 2]

# Selecting relevant columns
csv_ttops <- sf_ttops_4326[, c("treeID", "height", "lon", "lat")]

# export to csv
# write.csv(csv_ttops, paste0(root, ".\\6_ttop\\ttop.csv"), row.names = FALSE)

# Initialize counter
counter <- 0

# Define the file path for saving the shapefile
shp_file <- paste0(root, "/6_ttop/ttop.shp")

# Check if the file already exists
while (file.exists(shp_file)) {
  # If it does, increment the counter and update the file path
  counter <- counter + 1
  shp_file <- paste0(root, "/6_ttop/ttop_", counter, ".shp")
}

# Convert csv_ttops to sf object
sf_csv_ttops <- st_as_sf(csv_ttops, coords = c("lon", "lat"), crs = 4326)

# Write the sf object to a shapefile
st_write(sf_csv_ttops, shp_file)



#--------------------------------------------
# Outlining tree crowns
#-------------------------------------
# Create crown map
crowns_ras <- mcws(treetops = ttops, CHM = chm, minHeight = 2.0)

# Plot crowns
# plot(crowns_ras, col = sample(rainbow(50), nrow(unique(chm)), replace = TRUE), legend = FALSE, xlab = "", ylab = "", xaxt='n', yaxt = 'n')

# Create polygon crown map
crowns_poly <- mcws(treetops = ttops, CHM = chm, format = "polygons", minHeight = 2.0)

# Plot CHM
# plot(chm, xlab = "", ylab = "", xaxt='n', yaxt = 'n')

# Add crown outlines to the plot
# plot(crowns_poly$geometry, border = "blue", lwd = 0.5, add = TRUE)


# Convert crowns_poly to a data frame
crowns_df <- st_drop_geometry(crowns_poly)

# Merge attributes of sf_csv_ttops to crowns_df based on treeID
merged_crowns_df <- merge(crowns_df, sf_csv_ttops, by = "treeID", all.x = TRUE)

# Convert the merged data frame back to an sf object
crowns_poly_with_props <- st_sf(merged_crowns_df, geometry = crowns_poly$geometry)

# Calculate the area of each polygon and add it as a new column 'crown'
crowns_poly_with_props$crown <- st_area(crowns_poly_with_props)

# Initialize counter
counter <- 0

# Define the file path for saving the shapefile
crown_file <- paste0(root, "/7_crown/crowns_poly.shp")

# Check if the file already exists
while (file.exists(crown_file)) {
  # If it does, increment the counter and update the file path
  counter <- counter + 1
  crown_file <- paste0(root, "/7_crown/crowns_poly_", counter, ".shp")
}

# Write the merged sf object to a shapefile
st_write(crowns_poly_with_props, crown_file)



