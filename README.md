# Sindh River Flood Early Warning (research prototype)

A small website that shows whether the Indus river at three Sindh barrages
(Guddu, Sukkur, Kotri) will be **Normal**, on **Watch**, or at **Warning** level
over the next 24, 48 and 72 hours.

> Not an official warning. For real flood information use the Flood Forecasting
> Division (PMD) and PDMA Sindh.

## How it works (the big picture)

```
Open-Meteo (live river flow)
        |
   backend/live_data.py  --->  pipeline_lib.py + trained model  --->  backend/routes.py (JSON)
                                                                              |
                                                          static/ (browser page draws the result)
```

1. The server downloads recent river flow.
2. A model you trained earlier in a notebook predicts the level.
3. The web page asks the server for JSON and draws it.

## Folder guide

```
flood-warning/
├── main.py              START HERE - creates the app
├── pipeline_lib.py      data/model code shared with the training notebook (do not rename!)
├── requirements.txt     Python packages to install
├── backend/
│   ├── config.py          settings (paths, cache times)
│   ├── model_store.py     loads the trained model file
│   ├── live_data.py       downloads + remembers river flow
│   ├── routes.py          the API URLs (/api/forecast, /api/flow/..., ...)
│   └── about_content.py   text and test scores shown on the page
├── models/
│   └── tree24/48/72.joblib   your trained models go here (not included)
└── static/              everything the browser shows
    ├── index.html         page structure
    ├── css/style.css      colours and layout
    └── js/
        ├── utils.js         small helper functions
        ├── chart.js         draws the flow chart
        └── app.js           fetches data and fills in the page
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

You need `tree24.joblib`, `tree48.joblib` and `tree72.joblib` in `models/` (made by the training notebook). The first start also needs internet to calculate the cutoffs. Without it
the page opens but shows "Model file not found".

## Where to change things

| I want to...                        | Edit                      |
|-------------------------------------|---------------------------|
| Change cache time or model location | `backend/config.py`       |
| Add or change an API URL            | `backend/routes.py`       |
| Change the text under the scores    | `backend/about_content.py`|
| Change colours or layout            | `static/css/style.css`    |
| Change the chart                    | `static/js/chart.js`      |
| Add barrages or change features     | `pipeline_lib.py` (retrain the model afterwards) |

## Good to know

- `pipeline_lib.py` stays in the project root with that exact name: the saved model
  remembers it by name, and renaming or moving it breaks loading.
- Use the same scikit-learn version that trained the model (the page warns if different).
