from arduino_iot_cloud import ArduinoCloudClient
from arduino_secrets import DEVICE_ID, SECRET_KEY

from datetime import datetime
from collections import deque
import threading
import csv
import os
import time

import cv2
from flask import send_from_directory

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


# CONFIGURATION

# Length of one activity window (seconds)
WINDOW_SECONDS = 10

# Dash refresh rate (milliseconds)
UPDATE_INTERVAL = 200

# Folder for all collected data
DATA_FOLDER = "activity_data"

# Number of samples kept for the rolling live graph
MAX_LIVE_POINTS = 500

# Webcam settings
CAMERA_INDEX = 0
CAMERA_WARMUP_FRAMES = 10   # frames discarded when the camera starts
CAMERA_FLUSH_FRAMES = 3     # stale buffered frames discarded before each capture


os.makedirs(DATA_FOLDER, exist_ok=True)


# HELPERS

def now_string():
    """Full date-time with milliseconds, e.g. 2026-09-19 14:03:07.512"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def next_sequence_number():
    """
    Continue numbering from the highest sequence number already in
    DATA_FOLDER, so separate runs never produce duplicate numbers.
    """
    highest = 0

    for name in os.listdir(DATA_FOLDER):
        base, ext = os.path.splitext(name)

        if ext.lower() not in (".csv", ".jpg"):
            continue

        prefix = base.split("_")[0]

        if prefix.isdigit():
            highest = max(highest, int(prefix))

    return highest + 1


# LIVE DATA BUFFERS (rolling graph)

live_time = deque(maxlen=MAX_LIVE_POINTS)
live_x = deque(maxlen=MAX_LIVE_POINTS)
live_y = deque(maxlen=MAX_LIVE_POINTS)
live_z = deque(maxlen=MAX_LIVE_POINTS)


# CURRENT 10-SECOND WINDOW

window_time = []
window_x = []
window_y = []
window_z = []

# Protects data shared between Arduino Cloud, the collector and Dash
data_lock = threading.Lock()


# STATE

sequence_number = next_sequence_number()

latest_csv = ""
latest_image = ""

# The last saved window (shown in the "window graph")
last_window = None


# TEMPORARY CLOUD VALUES

x_value = None
y_value = None
z_value = None

x_received = False
y_received = False
z_received = False


# CAMERA

camera = None


def initialise_camera():
    global camera

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("WARNING: Laptop webcam could not be opened.")
        camera = None
        return

    # Keep the driver buffer small so captures are not stale
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # The first frames after opening are often dark while
    # auto-exposure settles, so discard them
    for _ in range(CAMERA_WARMUP_FRAMES):
        camera.read()
        time.sleep(0.05)

    print("Laptop webcam connected.")


def capture_activity_image(filename):

    if camera is None:
        print("No webcam available.")
        return False

    # Flush buffered (old) frames so the saved image is current
    for _ in range(CAMERA_FLUSH_FRAMES):
        camera.read()

    success, frame = camera.read()

    if not success:
        print("Could not capture webcam image.")
        return False

    return cv2.imwrite(filename, frame)


# SAVE 10-SECOND DATA WINDOW

def take_window():
    """
    Hand over the current window and start a new empty one.
    The lock is held only for this quick swap, so incoming
    samples are never blocked by file writing or the webcam.
    """
    global window_time, window_x, window_y, window_z

    with data_lock:

        if len(window_x) == 0:
            return None

        data = {
            "time": window_time,
            "x": window_x,
            "y": window_y,
            "z": window_z
        }

        window_time = []
        window_x = []
        window_y = []
        window_z = []

    return data


def save_activity_window():
    """Runs in the collector thread, outside the data lock."""

    global sequence_number
    global latest_csv
    global latest_image
    global last_window

    data = take_window()

    if data is None:
        print("No samples in this window, nothing saved.")
        return

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Matching filenames: <sequence>_<yyyymmddHHMMSS>
    base_name = f"{sequence_number}_{timestamp}"

    csv_filename = os.path.join(DATA_FOLDER, base_name + ".csv")
    image_filename = os.path.join(DATA_FOLDER, base_name + ".jpg")

    # --------------------------------------------------------
    # Save accelerometer data
    # --------------------------------------------------------

    with open(csv_filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["timestamp", "x", "y", "z"])

        for i in range(len(data["x"])):
            writer.writerow([
                data["time"][i],
                data["x"][i],
                data["y"][i],
                data["z"][i]
            ])

    # --------------------------------------------------------
    # Capture webcam image
    # --------------------------------------------------------

    image_saved = capture_activity_image(image_filename)

    if image_saved:
        print(f"Saved: {csv_filename} ({len(data['x'])} samples)")
        print(f"Saved: {image_filename}")
    else:
        print("CSV saved but webcam image was not saved.")

    # --------------------------------------------------------
    # Publish results for the dashboard
    # --------------------------------------------------------

    with data_lock:

        latest_csv = csv_filename
        latest_image = image_filename if image_saved else ""

        last_window = {
            "sequence": sequence_number,
            "time": data["time"],
            "x": data["x"],
            "y": data["y"],
            "z": data["z"]
        }

    sequence_number += 1


# ADD ACCELEROMETER SAMPLE

def add_sample(x, y, z):

    timestamp = now_string()

    with data_lock:

        # Rolling live graph
        live_time.append(timestamp)
        live_x.append(x)
        live_y.append(y)
        live_z.append(z)

        # Current 10-second window
        window_time.append(timestamp)
        window_x.append(x)
        window_y.append(y)
        window_z.append(z)


# COMBINE X, Y AND Z INTO ONE SAMPLE

def save_combined_data():

    global x_received
    global y_received
    global z_received

    if x_received and y_received and z_received:

        add_sample(x_value, y_value, z_value)

        print(
            f"X={x_value:.2f} | "
            f"Y={y_value:.2f} | "
            f"Z={z_value:.2f}"
        )

        x_received = False
        y_received = False
        z_received = False


# ARDUINO CLOUD CALLBACKS

def on_x(client, value):

    global x_value
    global x_received

    x_value = value
    x_received = True

    save_combined_data()


def on_y(client, value):

    global y_value
    global y_received

    y_value = value
    y_received = True

    save_combined_data()


def on_z(client, value):

    global z_value
    global z_received

    z_value = value
    z_received = True

    save_combined_data()


# 10-SECOND COLLECTION TIMER

def ten_second_collector():

    print("10-second activity collection started.")

    # Schedule against a fixed clock so saving time
    # does not make the windows drift
    next_tick = time.monotonic() + WINDOW_SECONDS

    while True:

        time.sleep(max(0, next_tick - time.monotonic()))
        next_tick += WINDOW_SECONDS

        try:
            save_activity_window()

        except Exception as error:
            # Keep the collector alive if a single save fails
            print(f"ERROR while saving window: {error}")


# ARDUINO IOT CLOUD

client = ArduinoCloudClient(
    device_id=DEVICE_ID,
    username=DEVICE_ID,
    password=SECRET_KEY
)

client.register("accelerometer_x", value=None, on_write=on_x)
client.register("accelerometer_y", value=None, on_write=on_y)
client.register("accelerometer_z", value=None, on_write=on_z)


# DASH APPLICATION

app = Dash(__name__)


# Dash only serves the "assets" folder automatically, so add a
# route that serves the saved activity images
@app.server.route("/activity_data/<path:filename>")
def serve_activity_file(filename):
    return send_from_directory(os.path.abspath(DATA_FOLDER), filename)


app.layout = html.Div([

    html.H1("Smartphone Activity Data Capture"),

    html.H3("Live Accelerometer Data (rolling)"),

    dcc.Graph(id="live-graph"),

    html.H3("Last Saved 10-Second Window and Activity Image"),

    html.Div(
        [
            html.Div(
                dcc.Graph(id="window-graph"),
                style={"flex": "1", "minWidth": "0"}
            ),

            html.Img(
                id="activity-image",
                style={
                    "width": "480px",
                    "height": "auto",
                    "marginLeft": "20px"
                }
            )
        ],
        style={"display": "flex", "alignItems": "flex-start"}
    ),

    html.Div(id="file-info", style={"marginTop": "20px"}),

    dcc.Interval(
        id="dash-update",
        interval=UPDATE_INTERVAL,
        n_intervals=0
    )

])


def make_figure(timestamps, x_data, y_data, z_data, title, height):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=timestamps, y=x_data, mode="lines", name="X"))
    fig.add_trace(go.Scatter(x=timestamps, y=y_data, mode="lines", name="Y"))
    fig.add_trace(go.Scatter(x=timestamps, y=z_data, mode="lines", name="Z"))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        xaxis_title="Time",
        yaxis_title="Acceleration",
        uirevision="live",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h")
    )

    return fig


# DASH UPDATE

@app.callback(
    [
        Output("live-graph", "figure"),
        Output("window-graph", "figure"),
        Output("activity-image", "src"),
        Output("file-info", "children")
    ],
    Input("dash-update", "n_intervals")
)
def update_dashboard(_):

    with data_lock:

        timestamps = list(live_time)
        x_data = list(live_x)
        y_data = list(live_y)
        z_data = list(live_z)

        current_image = latest_image
        current_csv = latest_csv
        window = last_window

    # --------------------------------------------------------
    # Rolling live graph
    # --------------------------------------------------------

    live_fig = make_figure(
        timestamps, x_data, y_data, z_data,
        "Live Smartphone Accelerometer",
        height=400
    )

    # --------------------------------------------------------
    # Graph of the last saved window (matches the saved files)
    # --------------------------------------------------------

    if window is not None:

        window_fig = make_figure(
            window["time"], window["x"], window["y"], window["z"],
            f"Saved window #{window['sequence']} "
            f"({len(window['x'])} samples)",
            height=360
        )

    else:

        window_fig = make_figure(
            [], [], [], [],
            "Waiting for the first 10-second capture...",
            height=360
        )

    # --------------------------------------------------------
    # Image path served by the Flask route above
    # --------------------------------------------------------

    if current_image:
        image_src = "/activity_data/" + os.path.basename(current_image)
    else:
        image_src = None

    # --------------------------------------------------------
    # File information
    # --------------------------------------------------------

    if current_csv:
        image_text = current_image if current_image else "not saved"
        file_text = (
            f"Latest data file: {current_csv} | "
            f"Latest image: {image_text}"
        )
    else:
        file_text = "Waiting for the first 10-second capture..."

    return live_fig, window_fig, image_src, file_text


# ARDUINO CLOUD THREAD

def start_cloud():

    print("Connecting to Arduino IoT Cloud...")

    client.start()


# START EVERYTHING

initialise_camera()

threading.Thread(target=start_cloud, daemon=True).start()

threading.Thread(target=ten_second_collector, daemon=True).start()

print(f"Next sequence number: {sequence_number}")
print("Open http://127.0.0.1:8050")

try:

    app.run(debug=False)

finally:

    if camera is not None:
        camera.release()

    cv2.destroyAllWindows()