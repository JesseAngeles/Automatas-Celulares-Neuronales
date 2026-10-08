import kagglehub

# Download latest version
path = kagglehub.dataset_download("brycehopkins/flame-3-nadir-thermal-plot-subset")

print("Path to dataset files:", path)