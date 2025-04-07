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

print(df)

# Create output directory for plots if it doesn't exist
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# Group data by TestName
test_names = df['TestName'].unique()

# Define metrics to plot
metrics = ['yolop_only']

# Set up the plot style
plt.style.use('seaborn-v0_8-darkgrid')
colors = sns.color_palette('Set2', len(metrics))

# Plot for each TestName separately
for test_name in test_names:
    test_data = df[df['TestName'] == test_name]
    
    # Create figure with appropriate size based on the amount of data
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.2
    x = np.arange(len(test_data['Weather'].unique()))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        plt.bar(x + i*bar_width, test_data[metric], width=bar_width, label=metric, color=colors[i])
    
    # Set labels, title and ticks
    plt.xlabel('Weather Condition')
    plt.ylabel('Count')
    plt.title(f'Comparison across Weather Conditions for {test_name}')
    plt.xticks(x + bar_width * (len(metrics) - 1) / 2, test_data['Weather'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{test_name}_comparison.png'))
    plt.close()

# Create a summary plot comparing all test names
plt.figure(figsize=(15, 8))

# For each metric, create a subplot
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    
    # Create a grouped bar plot for each test name
    sns.barplot(x='Weather', y=metric, hue='TestName', data=df)
    
    # Set labels and title
    plt.xlabel('Weather Condition')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Test Name')
    
    # Adjust layout
    plt.tight_layout()

# Adjust spacing between subplots
plt.subplots_adjust(top=0.9)
plt.suptitle('Summary Comparison Across All Tests', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the summary figure
plt.savefig(os.path.join(output_dir, 'all_tests_summary.png'))
plt.show()
