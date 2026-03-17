#!/bin/bash
# setup.sh - Automates the creation of the geo conda environment
#
# System dependency (install separately before running this script):
#   macOS:  brew install exiftool
#   Linux:  sudo apt install libimage-exiftool-perl

echo "Checking for Conda..."
if ! command -v conda &> /dev/null
then
    echo "Error: Conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

echo "Checking for exiftool..."
if ! command -v exiftool &> /dev/null
then
    echo "Warning: exiftool not found. Install it before using copy_gcp_images.py or TargetSighter."
    echo "  macOS:  brew install exiftool"
    echo "  Linux:  sudo apt install libimage-exiftool-perl"
fi

echo "Creating 'geo' environment..."
conda create -n geo -c conda-forge gdal flask tk pyproj scipy numpy opencv python=3.10 -y

echo ""
echo "Done! Activate the environment with:"
echo "    conda activate geo"
echo ""
echo "Verify key packages:"
echo "    conda activate geo && python -c \"from osgeo import gdal; import flask, pyproj, scipy, numpy, cv2; print('OK')\""
