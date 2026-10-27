"""Reusable pieces of the Sindh river flood early-warning pipeline.

Import this module BOTH in the training notebook and in the web service.
A saved scikit-learn pipeline stores its functions by name, so they must live
in an importable module (not in the notebook itself) or the model cannot be
loaded later.
"""
import pandas as pd
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

URL = "https://flood-api.open-meteo.com/v1/flood"

# Upstream to downstream.
LOCATIONS = {
    "Guddu": (28.42, 69.67),
    "Sukkur": (27.70, 68.86),
    "Kotri": (25.37, 68.26),
}

FEATURES = ["river_discharge", "change_1d", "change_3d", "mean_7d", "max_7d", "month"]
THRESHOLD = {"p90": 0.90, "p95": 0.95}          # lowest to highest
HORIZONS = {"24h": 1, "48h": 2, "72h": 3}       # days ahead
CLASS_LABELS = {0: "Normal", 1: "Watch", 2: "Warning"}


# ---------------------------------------------------------------- fetching
def fetch_data(url, params, name):
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    response = pd.DataFrame(response.json()["daily"])
    response["time"] = pd.to_datetime(response["time"])
    response["location"] = name
    return response


def fetch_all(start_date, end_date, locations=LOCATIONS, url=URL):
    """Fetch daily river discharge for every location and join into one table.
    Days without a value (the newest days can be empty) are dropped."""
    frames = []
    for name, (lat, lon) in locations.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge",
            "start_date": start_date,
            "end_date": end_date,
        }
        frames.append(fetch_data(url, params, name))
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["river_discharge"])
    return out.sort_values(["location", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------- pipeline steps
def feature_construction(data, sort_by, groups, target, change, roll):

    data = data.sort_values(sort_by).reset_index(drop=True)
    groupby = data.groupby(groups)[target]

    for i in change:
        data[f"change_{i}d"] = groupby.diff(i)

    for i in roll:
        data[f"mean_{i}d"] = groupby.rolling(i).mean().reset_index(level=0, drop=True)
        data[f"max_{i}d"] = groupby.rolling(i).max().reset_index(level=0, drop=True)

    data["month"] = data["time"].dt.month

    return data


def select_features(data):
    return data[FEATURES]


feature_step = FunctionTransformer(
    feature_construction,
    kw_args=dict(
        sort_by=["location", "time"],
        groups="location",
        target="river_discharge",
        change=[1, 2, 3],
        roll=[2, 3, 7],
    ),
)
select_step = FunctionTransformer(select_features)


def make_pipeline(model):
    """raw flow table -> features -> six clues -> trained model."""
    return Pipeline([
        ("features", feature_step),
        ("select", select_step),
        ("model", model),
    ])


# ---------------------------------------------------------------- training helpers
def data_split(data, time):
    train = data[data["time"] < time]
    test = data[~(data["time"] < time)]
    return train, test


def fit_cutoffs(data, groups, target, threshold):
    cut = data.groupby(groups)[target].quantile(list(threshold.values())).unstack()
    cut.columns = list(threshold.keys())
    return cut


def assign_severity(data, cut, groups, target, threshold):
    data = data.join(cut, on=groups)

    data["severity"] = 0

    for level, col in enumerate(threshold.keys(), start=1):
        data.loc[data[target] > data[col], "severity"] = level

    return data


# ---------------------------------------------------------------- live use
def check_window(raw, min_days=7):
    """The rolling features need an unbroken run of recent days. Returns a list of problems."""
    problems = []
    for name in LOCATIONS:
        dates = raw.loc[raw["location"] == name, "time"].sort_values()
        if len(dates) < min_days:
            problems.append(f"{name}: only {len(dates)} days of data (need {min_days}).")
            continue
        last = dates.tail(min_days)
        if not (last.diff().dropna() == pd.Timedelta(days=1)).all():
            problems.append(f"{name}: days are missing in the last {min_days} days.")
    return problems


def predict_latest(raw, pipelines):
    """Newest-day prediction per location from every pipeline."""
    out = feature_step.transform(raw)[["location", "time"]]
    for name, p in pipelines.items():
        out[f"pred_{name}"] = p.predict(raw)
    return out.sort_values("time").groupby("location").tail(1).reset_index(drop=True)


def current_status(raw, cut):
    """Class of the newest day's flow per location, using the saved cutoffs."""
    latest = raw.sort_values("time").groupby("location").tail(1)
    labelled = assign_severity(latest, cut, "location", "river_discharge", THRESHOLD)
    return labelled[["location", "time", "river_discharge", "severity"]].reset_index(drop=True)
