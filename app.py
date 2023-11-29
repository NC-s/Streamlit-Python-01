
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
import pandas as pd
import numpy as np

file_name = "app.py"
server_port = "8502"
DATA_URL = (
    # "https://s3-us-west-2.amazonaws.com/"
    # "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
    "streamlit-demo-data/Motor_Vehicle_Collisions_-_Crashes.csv"
)


# Your streamlit code
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