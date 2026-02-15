#!/bin/bash
# setup_env.sh - Automates the creation of the gdal_env conda environment

echo "Checking for Conda..."
if ! command -v conda &> /dev/null
then
    echo "Error: Conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

echo "Creating 'gdal' environment..."
conda create -n gdal -c conda-forge gdal flask tk python=3.10 -y

echo "Done! Activate the environment with:"
echo "    conda activate gdal"
