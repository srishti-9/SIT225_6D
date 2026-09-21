"""
StreamBuffer is a generic rolling buffer for smooth Plotly Dash live updates.

This is the Q2 wrapper function/class referenced in the report. It is a
generalised version of the accelerometer-only add_sample(x, y, z) used in
smooth_dash.py: instead of being hard-coded to three channels, it accepts
any number of named channels, so it can be reused for any live-streamed
data (a single temperature sensor, five strain-gauge channels, etc.).

It is NOT wired into smooth_dash.py by default — that script still uses
its own add_sample(x, y, z) to match the submitted demonstration video.
This file exists purely as the documented, reusable API described in Q2.
"""

from collections import deque
from datetime import datetime
import threading


class StreamBuffer:
    """
    Generic rolling buffer for smooth Plotly Dash live updates.

    Wraps one or more named data channels (e.g. 'x', 'y', 'z', or
    'temperature', 'humidity', ...) in fixed-length deques, plus a
    matching timestamp deque, so any producer (cloud callback,
    serial read, sensor poll, ...) can push samples on arrival and
    any dcc.Interval callback can read the latest window without
    the two ever needing to run at the same rate.
    """

    def __init__(self, channel_names, maxlen=200, time_format="%H:%M:%S.%f"):
        self.channel_names = list(channel_names)
        self.maxlen = maxlen
        self.time_format = time_format
        self.time_buffer = deque(maxlen=maxlen)
        self.buffers = {name: deque(maxlen=maxlen) for name in self.channel_names}
        self._lock = threading.Lock()

    def add_sample(self, timestamp=None, **channel_values):
        """Append one synchronised sample across all given channels.

        timestamp : datetime or None. If None, uses datetime.now().
        channel_values : e.g. add_sample(x=1.2, y=-0.3, z=9.8)
        """
        t = (timestamp or datetime.now()).strftime(self.time_format)[:-3]
        with self._lock:
            self.time_buffer.append(t)
            for name in self.channel_names:
                self.buffers[name].append(channel_values.get(name))

    def get_trace(self, name):
        """Return (list_of_timestamps, list_of_values) for one channel,
        safe to call from a dcc.Interval callback at any time."""
        with self._lock:
            return list(self.time_buffer), list(self.buffers[name])