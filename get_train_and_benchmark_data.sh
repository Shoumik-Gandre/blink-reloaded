#!/bin/bash

benchmark_data_folder="train_and_benchmark_data"
if [[ ! -d $benchmark_data_folder ]]; then
    mkdir -p $benchmark_data_folder
fi

fileid="1IDjXFnNnHf__MO5j_onw4YwR97oS8lAy"
filename="train_and_benchmark_data.zip"

# Get the confirmation token and download the file
curl "https://drive.usercontent.google.com/download?id=${fileid}&confirm=xxx" -o ${filename}

# Unzip the file
unzip -d $benchmark_data_folder $filename

# Move the contents
mv $benchmark_data_folder/data/* $benchmark_data_folder/
rm -r $benchmark_data_folder/data/
