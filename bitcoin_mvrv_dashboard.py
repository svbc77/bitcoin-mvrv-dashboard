import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from prophet import Prophet

# Carica dati storici
df = pd.read_csv("bitcoin_mvrv_sample.csv")
df["date"] = pd.to_datetime(df["date"])

# Modello Prophet per MVRV totale
prophet_df = df[["date", "mvrv_total"]].rename(columns={"date": "ds", "mvrv_total": "y"})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Bitcoin MVRV Dashboard (2021 – Oggi)"),
    dcc.Graph(
        id="mvrv-graph",
        figure={
            "data": [
                go.Scatter(x=df["date"], y=df["mvrv_total"], mode="lines", name="MVRV Totale"),
                go.Scatter(x=df["date"], y=df["sth_mvrv"], mode="lines", name="STH-MVRV"),
                go.Scatter(x=df["date"], y=df["lth_mvrv"], mode="lines", name="LTH-MVRV"),
            ],
            "layout": go.Layout(title="Andamento MVRV 2021–Oggi", xaxis={"title": "Data"}, yaxis={"title": "MVRV Ratio"})
        }
    ),
    dcc.Graph(
        id="supply-ratio",
        figure={
            "data": [
                go.Scatter(x=df["date"], y=df["sth_ratio"], mode="lines", name="STH Supply %"),
                go.Scatter(x=df["date"], y=df["lth_ratio"], mode="lines", name="LTH Supply %"),
            ],
            "layout": go.Layout(title="Distribuzione Supply LTH vs STH", xaxis={"title": "Data"}, yaxis={"title": "%"})
        }
    ),
    dcc.Graph(
        id="mvrv-forecast",
        figure={
            "data": [
                go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Previsione MVRV Totale")
            ],
            "layout": go.Layout(title="Previsione MVRV (Prossimi 6 mesi)", xaxis={"title": "Data"}, yaxis={"title": "MVRV Predetto"})
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
