# Lane Departure Warning System for CARLA

A vision-based Lane Departure Warning (LDW) system implemented in the CARLA autonomous driving simulator, using YOLOPv2 for real-time lane detection.

## Features

- Real-time lane detection using YOLOPv2 model
- Lane departure warning generation
- Support for multiple weather and lighting conditions
- Record and playback system for reproducible testing
- MQTT event publishing for system integration
- Performance logging and analysis
- Comparison with CARLA's ground truth lane invasion detection

## System Requirements

### Hardware
- NVIDIA GPU (recommended for optimal performance)
- Minimum 8GB RAM
- 20GB free disk space

### Software
- Windows 10/11 or Linux (Ubuntu 18.04+)
- Python 3.7
- CARLA 0.9.15 simulator
- HiveMQ account (for MQTT integration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/TonelliLuca/Lane_Departure_Warning_CARLA.git
   cd Lane_Departure_Warning_CARLA
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate carla-env
   ```

3. Download the YOLOPv2 model:
   ```bash
   mkdir -p data/weights
   # Download manually from https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
   # and place in data/weights/ directory
   ```

4. Configure MQTT credentials:
   Create a `.env` file in the project root with:
   ```
   HIVE_MQ_USERNAME=your_username
   HIVE_MQ_PASSWORD=your_password
   ```

## Usage

### Running the LDW System

The simplest way to start the system:
```bash
python launcher.py
```

## Performance

The lane detection model demonstrates efficient real-time performance:
- Initialization overhead: ~1.46s for first frame
- Steady-state performance: 43.7ms - 61.5ms per frame
- Typical processing range: 45-55ms per frame
- Frame rate: 16-22 FPS during normal operation

## Authors

- Alessandro Becci - alessandro.becci@studio.unibo.it - [GitHub](https://github.com/stormtroober)
- Luca Tonelli - luca.tonelli11@studio.unibo.it - [GitHub](https://github.com/TonelliLuca)

[Project Pages](https://tonelliluca.github.io/Lane_Departure_Warning_CARLA/)

University of Bologna - Master Degree in Computer Science and Engineering