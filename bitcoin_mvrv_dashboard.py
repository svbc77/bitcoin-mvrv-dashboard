import dash
from dash import html, dcc
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet

# Caricamento dati CSV
df = pd.read_csv("bitcoin_mvrv_sample.csv")
df["date"] = pd.to_datetime(df["date"])

# Previsione MVRV totale
prophet_df = df[["date", "mvrv_total"]].rename(columns={"date": "ds", "mvrv_total": "y"})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

# App Dash
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Bitcoin MVRV Dashboard"),
    dcc.Graph(figure={
        "data": [
            go.Scatter(x=df["date"], y=df["mvrv_total"], name="MVRV Totale"),
            go.Scatter(x=df["date"], y=df["sth_mvrv"], name="STH-MVRV"),
            go.Scatter(x=df["date"], y=df["lth_mvrv"], name="LTH-MVRV"),
        ],
        "layout": go.Layout(title="Andamento MVRV")
    }),
    dcc.Graph(figure={
        "data": [
            go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Previsione MVRV")
        ],
        "layout": go.Layout(title="Previsione 6 mesi")
    })
])
