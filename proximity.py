from dataclasses import dataclass
from collections import deque

@dataclass
class ProximityEvent:
    level: str          # "approaching" | "danger"
    message: str
    rel_area: float

class ProximityLogic:
    def __init__(
        self,
        window=10,
        approaching_area=0.06,
        danger_area=0.18,          # bump this up if you want "danger" closer
        approach_slope=0.002,
        cooldown_frames=25,
        danger_cooldown_frames=80, # prevents danger spam
        danger_release=0.16,       # must back off below this to re-arm danger
        min_delta=0.02             # total growth over window to count as "approaching"
    ):
        self.window = window
        self.areas = deque(maxlen=window)

        self.approaching_area = approaching_area
        self.danger_area = danger_area
        self.danger_release = danger_release

        self.approach_slope = approach_slope
        self.min_delta = min_delta

        self.cooldown_frames = cooldown_frames
        self.cooldown = 0

        self.danger_cooldown_frames = danger_cooldown_frames
        self.danger_cooldown = 0

        self.last_label = None
        self.state = "idle"  # idle -> approaching_announced -> danger

    def _slope(self):
        if len(self.areas) < self.window:
            return 0.0
        return (self.areas[-1] - self.areas[0]) / (self.window - 1)

    def update(self, label: str, rel_area: float):
        # object changed => reset
        if self.last_label != label:
            self.last_label = label
            self.areas.clear()
            self.state = "idle"
            self.cooldown = 0
            self.danger_cooldown = 0

        self.areas.append(rel_area)

        if self.cooldown > 0:
            self.cooldown -= 1
        if self.danger_cooldown > 0:
            self.danger_cooldown -= 1

        if len(self.areas) < self.window:
            return None

        slope = self._slope()
        delta = self.areas[-1] - self.areas[0]
        sm = sum(self.areas) / len(self.areas)

        # DANGER: sticky + cooldown
        if sm >= self.danger_area:
            if self.state != "danger" and self.danger_cooldown == 0:
                self.state = "danger"
                self.danger_cooldown = self.danger_cooldown_frames
                return ProximityEvent("danger", f"dangerously close to {label} please turn", sm)
            return None

        # release danger only after backing off
        if self.state == "danger" and sm < self.danger_release:
            self.state = "idle"
            self.cooldown = self.cooldown_frames
            return None

        # APPROACHING: only once per approach session
        if self.state == "idle" and self.cooldown == 0:
            if sm >= self.approaching_area and slope >= self.approach_slope and delta >= self.min_delta:
                self.state = "approaching_announced"
                self.cooldown = self.cooldown_frames
                return ProximityEvent("approaching", f"approaching {label}", sm)

        # re-arm "approaching" if they back off enough
        if self.state == "approaching_announced" and sm < (self.approaching_area - 0.02):
            self.state = "idle"

        return None