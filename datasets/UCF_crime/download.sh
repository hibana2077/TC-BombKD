#!/bin/bash

# Suggested to run in shell or bash
echo "Starting UCF-Crime dataset download script..."

# Create download directory
echo "Creating download directory: ~/datasets/UCF-Crime"
mkdir -p ~/datasets/UCF-Crime

# Change to the directory
echo "Changing to download directory..."
cd ~/datasets/UCF-Crime

# Download the dataset zip file
echo "Downloading UCF_Crimes.zip from http://crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
wget --continue --tries=3 --timeout=30 http://crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip -O UCF-Crime.zip

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed. Please check your internet connection and try again."
    exit 1
fi

# Unzip the file
echo "Unzipping UCF-Crime.zip..."
unzip UCF-Crime.zip

# Check if unzip was successful
if [ $? -eq 0 ]; then
    echo "Unzipping completed successfully."
    echo "Dataset is ready in ~/datasets/UCF-Crime"
else
    echo "Unzipping failed. Please check the zip file integrity."
    exit 1
fi