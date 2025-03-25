from datetime import datetime


class DetectionLogger:
    def __init__(self):
        self.detections = []  # [(timestamp, detector_name, detection_status, confidence)]
        self.agreement_count = 0
        self.total_events = 0

    def log_detection(self, detector_name, status, confidence=1.0):
        timestamp = datetime.now()
        self.detections.append((timestamp, detector_name, status, confidence))

    def get_stats(self):
        if not self.detections:
            return {"agreement_rate": 0, "events": 0}

        # Group detections by time window (1 second)
        time_windows = {}
        for detection in self.detections:
            timestamp = detection[0]
            key = timestamp.strftime("%Y%m%d%H%M%S")  # 1-second window
            if key not in time_windows:
                time_windows[key] = {"yolop": False, "carla": False}

            if detection[1] == "YOLOP":
                time_windows[key]["yolop"] = detection[2]  # True for detected crossing
            elif detection[1] == "CARLA":
                time_windows[key]["carla"] = detection[2]

        # Calculate agreement
        agreements = sum(1 for window in time_windows.values()
                         if window["yolop"] == window["carla"])

        return {
            "agreement_rate": agreements / len(time_windows),
            "events": len(time_windows),
            "yolop_only": sum(1 for w in time_windows.values() if w["yolop"] and not w["carla"]),
            "carla_only": sum(1 for w in time_windows.values() if w["carla"] and not w["yolop"])
        }