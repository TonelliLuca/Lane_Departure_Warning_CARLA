import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_log_file(log_path):
    """Parse the frame performance log file."""
    data = []
    frame_data = {}

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Start of new frame or end of file
            if '----' in line:
                if frame_data:
                    data.append(frame_data)
                    frame_data = {}
                continue

            # Extract time
            time_match = re.search(r'Time taken for image processing and analysis: ([\d\.]+) seconds', line)
            if time_match:
                frame_data['processing_time'] = float(time_match.group(1))
                continue

            # Extract image dimensions
            dim_match = re.search(r'Image Height: (\d+), Image Width: (\d+)', line)
            if dim_match:
                frame_data['height'] = int(dim_match.group(1))
                frame_data['width'] = int(dim_match.group(2))
                continue

            # Extract processed image size
            size_match = re.search(r'Image Size \(after processing\): \((\d+), (\d+), (\d+)\)', line)
            if size_match:
                frame_data['channels'] = int(size_match.group(1))
                frame_data['processed_height'] = int(size_match.group(2))
                frame_data['processed_width'] = int(size_match.group(3))
                continue

            # Extract max value
            max_match = re.search(r'Max value in processed image: (\d+)', line)
            if max_match:
                frame_data['max_value'] = int(max_match.group(1))
                continue

            # Extract result
            result_match = re.search(r'Result: (\w+)', line)
            if result_match:
                frame_data['result'] = result_match.group(1)
                continue

            # Extract crossing
            crossing_match = re.search(r'Crossing: (\w+)', line)
            if crossing_match:
                frame_data['crossing'] = crossing_match.group(1)
                continue

    # Add the last frame if it exists
    if frame_data:
        data.append(frame_data)

    return pd.DataFrame(data)

def analyze_and_visualize(df, output_path):
        """Analyze the data and create visualizations."""
        # Print statistics to verify data
        print(f"Data shape: {df.shape}")
        print("Column stats:")
        print(df.describe())

        plt.figure(figsize=(15, 10))

        # First graph: Processing Time Distribution
        plt.subplot(2, 1, 1)
        df['processing_time'] = pd.to_numeric(df['processing_time'], errors='coerce')
        sns.histplot(df['processing_time'], kde=True)
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')

        # Second graph: Variance of Analysis Time Over Logs
        plt.subplot(2, 1, 2)
        df['log_index'] = range(len(df))
        df['processing_time_variance'] = df['processing_time'].expanding().var()
        sns.lineplot(x='log_index', y='processing_time_variance', data=df)
        plt.title('Variance of Analysis Time Over Logs')
        plt.xlabel('Log Index')
        plt.ylabel('Variance of Processing Time')

        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Analysis saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze frame performance data.')
    parser.add_argument('--input', required=True, help='Path to the log file')
    parser.add_argument('--output', required=True, help='Path for the output visualization')
    args = parser.parse_args()

    try:
        df = parse_log_file(args.input)
        print(f"Successfully loaded {len(df)} frame entries")
        analyze_and_visualize(df, args.output)
    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    main()