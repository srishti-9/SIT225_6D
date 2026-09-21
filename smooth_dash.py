from arduino_iot_cloud import ArduinoCloudClient
from arduino_secrets import DEVICE_ID, SECRET_KEY
from collections import deque
from datetime import datetime
import threading
import csv
import os

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# CONFIG


CSV_FILE = "accelerometer_xyz.csv"
MAX_POINTS = 200

# Rolling buffers
time_buffer = deque(maxlen=MAX_POINTS)
x_buffer = deque(maxlen=MAX_POINTS)
y_buffer = deque(maxlen=MAX_POINTS)
z_buffer = deque(maxlen=MAX_POINTS)

# Temporary values from cloud
x_value = None
y_value = None
z_value = None

x_received = False
y_received = False
z_received = False


# CSV


if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z"])


# Wrapper function (Q2)


def add_sample(x, y, z):
    """
    Smooth streaming wrapper.

    Stores the latest accelerometer values inside
    rolling buffers for Plotly Dash.
    """

    t = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    time_buffer.append(t)
    x_buffer.append(x)
    y_buffer.append(y)
    z_buffer.append(z)

# Save data


def save_combined_data():

    global x_received, y_received, z_received

    if x_received and y_received and z_received:

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, x_value, y_value, z_value])

        add_sample(x_value, y_value, z_value)

        print(
            f"{timestamp} | "
            f"X={x_value:.2f} "
            f"Y={y_value:.2f} "
            f"Z={z_value:.2f}"
        )

        x_received = False
        y_received = False
        z_received = False


# Cloud callbacks


def on_x(client, value):
    global x_value, x_received
    x_value = value
    x_received = True
    save_combined_data()

def on_y(client, value):
    global y_value, y_received
    y_value = value
    y_received = True
    save_combined_data()

def on_z(client, value):
    global z_value, z_received
    z_value = value
    z_received = True
    save_combined_data()

# Arduino Cloud


client = ArduinoCloudClient(
    device_id=DEVICE_ID,
    username=DEVICE_ID,
    password=SECRET_KEY
)

client.register(
    "accelerometer_x",
    value=None,
    on_write=on_x
)

client.register(
    "accelerometer_y",
    value=None,
    on_write=on_y
)

client.register(
    "accelerometer_z",
    value=None,
    on_write=on_z
)


# Dash App


app = Dash(__name__)

app.layout = html.Div([

    html.H2("Live Smartphone Accelerometer"),

    dcc.Graph(id="live-graph"),

    dcc.Interval(
        id="interval",
        interval=50,
        n_intervals=0
    )

])

@app.callback(
    Output("live-graph", "figure"),
    Input("interval", "n_intervals")
)

def update_graph(_):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(time_buffer),
        y=list(x_buffer),
        mode="lines",
        name="X"
    ))

    fig.add_trace(go.Scatter(
        x=list(time_buffer),
        y=list(y_buffer),
        mode="lines",
        name="Y"
    ))

    fig.add_trace(go.Scatter(
        x=list(time_buffer),
        y=list(z_buffer),
        mode="lines",
        name="Z"
    ))

    fig.update_layout(

        template="plotly_dark",

        height=500,

        xaxis_title="Time",

        yaxis_title="Acceleration",

        uirevision=True,

        margin=dict(l=40, r=20, t=40, b=40),

        legend=dict(
            orientation="h",
            y=1.1
        )

    )

    return fig


# Run both simultaneously


def start_cloud():
    print("Connecting to Arduino IoT Cloud...")
    client.start()

threading.Thread(
    target=start_cloud,
    daemon=True
).start()

print("Open http://127.0.0.1:8050")

app.run(debug=False)