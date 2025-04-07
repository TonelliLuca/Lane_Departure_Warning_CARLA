import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import os

# Function to parse the results string into a dictionary
def parse_results(results_str):
    try:
        # Extract dictionary part with regex pattern for better reliability
        match = re.search(r'\{.*?\}', results_str)
        if match:
            dict_str = match.group(0)
            # Parse the dictionary string safely
            result_dict = eval(dict_str)
            return result_dict
        return {}
    except Exception as e:
        print(f"Error parsing results: {results_str}")
        print(f"Error: {e}")
        return {}

# Function to parse the log lines into a DataFrame
def parse_log_data(log_data):
    data = []
    for line in log_data.strip().split('\n'):
        # Skip empty lines or lines starting with ???
        if not line or line.startswith('???'):
            continue
        
        try:
            # Split by semicolon and trim whitespace
            parts = [part.strip() for part in line.split(';')]
            
            # Create basic entry structure
            entry = {}
            
            # Process each part to extract key-value pairs
            for part in parts:
                if ': ' in part:
                    key, value = part.split(': ', 1)
                    if key == 'Results':
                        # Parse the Results section separately
                        results_dict = parse_results(value)
                        # Add results dictionary items to entry
                        for k, v in results_dict.items():
                            entry[k] = v
                    else:
                        entry[key] = value
            
            data.append(entry)
        except Exception as e:
            print(f"Error parsing line: {line}")
            print(f"Error: {e}")
    
    # Convert numeric columns to appropriate types
    df = pd.DataFrame(data)
    numeric_cols = ['events', 'yolop_only', 'carla_only', 'agreements', 'PlaybackIndex']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load log data from an external file
log_data_file = 'log_data.txt'
with open(log_data_file, 'r') as file:
    log_data = file.read()

# Parse the log data
df = parse_log_data(log_data)

# Create output directory for plots if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Get unique test names
test_names = df['TestName'].unique()

# Create a bar chart for each test name
for test_name in test_names:
    # Filter data for the current test name
    test_data = df[df['TestName'] == test_name]
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    weathers = test_data['Weather'].tolist()
    width = 0.25  # Width of the bars
    indices = np.arange(len(weathers))  # X locations for groups
    
    # Create bars for each metric
    yolop_bars = plt.bar(indices - width, test_data['yolop_only'], width, label='YOLOP Only')
    carla_bars = plt.bar(indices, test_data['carla_only'], width, label='CARLA Only')
    agreement_bars = plt.bar(indices + width, test_data['agreements'], width, label='Agreements')
    
    # Add labels, title, and legend
    plt.xlabel('Weather Conditions')
    plt.ylabel('Count')
    plt.title(f'Lane Departure Detection Comparison for Test: {test_name}')
    plt.xticks(indices, weathers, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Add values on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    add_labels(yolop_bars)
    add_labels(carla_bars)
    add_labels(agreement_bars)
    
    # Save the plot
    plt.savefig(f'{output_dir}/{test_name}_comparison.png')
    plt.close()

print(f"Bar charts created for each test name and saved in '{output_dir}' directory.")