import numpy as np
import pandas as pd


class Profiler(object):
    def __init__(self):
        self.data = {}

    def add_time(self, event_name, time_elapsed):
        if event_name not in self.data:
            self.data[event_name] = 0
        self.data[event_name] += time_elapsed

    def export(self):
        return pd.DataFrame(
            list(self.data.items()), columns=["event_name", "time_elapsed"]
        )
