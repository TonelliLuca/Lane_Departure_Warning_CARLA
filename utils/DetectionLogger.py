from datetime import datetime, timedelta


class DetectionLogger:
    def __init__(self, agreement_window_seconds=2):
        self.detections = []  # [(timestamp, detector_name, detection_status, confidence)]
        self.agreement_window = timedelta(seconds=agreement_window_seconds)
        self.last_yolop_crossing = None
        self.last_carla_crossing = None

    def log_detection(self, detector_name, status, confidence=1.0):
        timestamp = datetime.now()
        self.detections.append((timestamp, detector_name, status, confidence))

        # Track the latest crossing events for each detector
        if detector_name == "YOLOP" and status:
            self.last_yolop_crossing = timestamp
        elif detector_name == "CARLA" and status:
            self.last_carla_crossing = timestamp

    def get_stats(self):
        if not self.detections:
            return {"agreement_rate": 0, "events": 0, "yolop_only": 0, "carla_only": 0}

        # Group detections into crossing events with wider time windows
        crossing_events = []
        current_event = {"start": None, "end": None, "yolop": False, "carla": False}

        # Sort detections by timestamp
        sorted_detections = sorted(self.detections, key=lambda x: x[0])

        for timestamp, detector, status, _ in sorted_detections:
            # Only consider positive crossing detections
            if not status:
                continue

            if current_event["start"] is None:
                # Start a new event
                current_event = {
                    "start": timestamp,
                    "end": timestamp,
                    "yolop": detector == "YOLOP",
                    "carla": detector == "CARLA"
                }
            elif timestamp - current_event["end"] > self.agreement_window:
                # This detection is beyond our time window, save the current event and start a new one
                crossing_events.append(current_event)
                current_event = {
                    "start": timestamp,
                    "end": timestamp,
                    "yolop": detector == "YOLOP",
                    "carla": detector == "CARLA"
                }
            else:
                # This detection belongs to the current event
                current_event["end"] = timestamp
                if detector == "YOLOP":
                    current_event["yolop"] = True
                else:
                    current_event["carla"] = True

        # Add the last event if it exists
        if current_event["start"] is not None:
            crossing_events.append(current_event)

        if not crossing_events:
            return {"agreement_rate": 0, "events": 0, "yolop_only": 0, "carla_only": 0}

        agreements = sum(1 for event in crossing_events if event["yolop"] and event["carla"])
        yolop_only = sum(1 for event in crossing_events if event["yolop"] and not event["carla"])
        carla_only = sum(1 for event in crossing_events if event["carla"] and not event["yolop"])

        return {
            "events": len(crossing_events),
            "yolop_only": yolop_only,
            "carla_only": carla_only,
            "agreements": agreements
        }