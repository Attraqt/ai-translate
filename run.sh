#!/bin/bash

# 1. Download and install Python, pip, and pipenv
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install necessary Python libraries
pip install torch torchvision pandas tqdm transformers sentencepiece datasets

# 4. Ask for the CSV URL and download the file
echo "Enter the URL of the CSV file to download:"
read csv_url
filename="dataset.csv"
wget -O "$filename" "$csv_url"

# 5. Ask for the locale
echo "Enter the locale (e.g., en, fr, es):"
read locale

# 6. Launch the Python script
output_file="output-$(date +%Y%m%d%H%M%S).csv"
python translate-datasets.py "$filename" "$output_file" "$locale"

echo "Translation complete. Output saved to $output_file"

# 7. Zip the output file
zip "${output_file}.zip" "$output_file"
echo "Output file zipped as ${output_file}.zip"

# 8. Start a simple HTTP server for download
python3 -m http.server 8000 &
server_pid=$!
echo "Download the output file from: http://$(hostname -I | awk '{print $1}'):8000/${output_file}.zip"

# 9. Instructions for stopping the server
echo "To stop the server, run: kill $server_pid"
