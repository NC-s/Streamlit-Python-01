
# Advantage for Streamlit

# Web frameworks like flask, you'll understand that unless you write
# specific code to implement some sort of cacheing mechanism

# Each update of the slider involves rerunning the entire script, meaning
# you'll be loading the 1.67 million rows of data over and over again, which
# can really put a dent in performance.

# But with stream lit, we can use a simple function decorator
# to intelligently cache the data

# Unless the input to the data or the data itself has been modified, the app
# will use the cache data over and over again to perform your computations

import sys
import subprocess
import streamlit as st
from streamlit.web import cli as stcli
from streamlit import runtime as st_runtime
from streamlit_jupyter import StreamlitPatcher, tqdm
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px

file_name = "app.py"
server_port = "8502"
DATA_URL = (
    # "https://s3-us-west-2.amazonaws.com/"
    "streamlit-demo-data/Motor_Vehicle_Collisions_-_Crashes.csv"
)

StreamlitPatcher().jupyter()  # register streamlit with jupyter-compatible wrappers


def main():
    st.title("Streamlit: Caching")
    st.markdown("### The is the first dashboard")

    @st.cache_data(persist=True)
    def load_data(nrows):
        data = pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[
            ['CRASH_DATE', 'CRASH_TIME']])
        data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
        def lowercase(x): return str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data.rename(
            columns={'crash_date_crash_time': 'date/time'}, inplace=True)
        return data

    data = load_data(100000)
    # Keep the Unfiltered data for later use
    original_data = data

    st.header("Where are the most people injured in NYC?")
    injured_people = st.slider(
        "Number of persons injured in vehicle collisions", 0, 19)
    st.map(data.query("injured_persons >= @injured_people")
           [["latitude", "longitude"]].dropna(how="any"))

    st.header("How many collisions occur during a given time of day?")
    hour = st.slider("Hour to look at", 0, 23)

    # If we need a sidebar for the slider
    # hour = st.sidebar.slider("Hour to look at", 0, 23)
    data = data[data['date/time'].dt.hour == hour]

    st.markdown("Vehicle collisions between %i:00 and %i:00" %
                (hour, (hour + 1) % 24))
    midpoint = (np.average(data['latitude']), np.average(data['longitude']))

    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data[['date/time', 'latitude', 'longitude']],
                get_position=['longitude', 'latitude'],
                radius=100,  # The size of the bar chart one the map
                extruded=True,  # The bar chart is extruded (3D)
                pickable=True,
                elevation_scale=4,
                elevation_range=[0, 1000],
            ),
        ],
    ))

    # Using plotly
    st.subheader("Breakdown by minute between %i:00 and %i:00" %
                 (hour, (hour + 1) % 24))

    filtered = data[
        (data['date/time'].dt.hour >=
         hour) & (data['date/time'].dt.hour < (hour + 1))
    ]

    hist = np.histogram(
        filtered['date/time'].dt.minute, bins=60, range=(0, 60))[0]  # 60 Minutes for an hour
    # Crashes data from histogram
    chart_data = pd.DataFrame({'minute': range(60), 'crashes': hist})
    fig = px.bar(chart_data, x='minute', y='crashes',
                 hover_data=['minute', 'crashes'], height=400)
    st.write(fig)

    # Align HTML through Caption
    st.caption(
        "<p style='text-align: right; color: grey;'>*Allow to zoom in by left click, Data Source: NYC Motor Vehicle Collisions</style>", unsafe_allow_html=True)

    # Using pandas data option
    st.header("Top 5 dangerous streets by affected type")
    # Drop Down Menu
    select = st.selectbox(
        "Affected type of people", ['Pedestrians', 'Cyclists', 'Motorists'])

    if select == "Pedestrians":
        # Only injured pedestrians & Descending order & dropna for any missing data & Top 5 dangerous streets
        st.write(original_data.query("injured_pedestrians >= 1")[[
                 "on_street_name", "injured_pedestrians"]].sort_values(by=['injured_pedestrians'], ascending=False).dropna(how='any')[:5])
    elif select == "Cyclists":
        # Only injured cyclists & Descending order & dropna for any missing data & Top 5 dangerous streets
        st.write(original_data.query("injured_cyclists >= 1")[[
                 "on_street_name", "injured_cyclists"]].sort_values(by=['injured_cyclists'], ascending=False).dropna(how='any')[:5])
    elif select == "Motorists":
        # Only injured motorists & Descending order & dropna for any missing data & Top 5 dangerous streets
        st.write(original_data.query("injured_motorists >= 1")[[
                 "on_street_name", "injured_motorists"]].sort_values(by=['injured_motorists'], ascending=False).dropna(how='any')[:5])

    # Raw Data Section
    if st.checkbox("Show Raw Data", False):
        st.subheader("Raw Data")
        st.write(data)


if __name__ == '__main__':
    if st_runtime.exists():
        main()
    else:
        subprocess.call(["streamlit", "run", file_name,
                        "--server.port", f"{server_port}"])
