import splitfolders

input_folder = "dataset"  # This is your original dataset folder
output_folder = "output_dataset"  # This folder will contain the split data
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.7, 0.15, 0.15))