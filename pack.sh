#!/bin/bash

# Create a mock weights directory if it doesn't exist
mkdir -p weights

# Create a flat zip archive without a wrapper folder
zip -r submission.zip Dockerfile solution.py weights/

echo "Packaging complete! submission.zip is ready for upload."
