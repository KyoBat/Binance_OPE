
from datetime import datetime, timedelta
from pickle import NONE
from dash import dcc, html
import dash
from dash import dcc
from dash import dash_table

import dash_core_components as dcc
import dash_html_components as html
from dash.dash_table.Format import Group
from dash.dependencies import Output, Input,State
import pandas as pd
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import httpx
from starlette.responses import JSONResponse
from prophet import Prophet
import pytz

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app_binance = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app_binance.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    html.H1('Les prédictions pour binance', style={'color': 'aquamarine', 'textAlign': 'center'}),
    html.Button(dcc.Link('Evolution cryptocurrency', href='/cryptocurrency')),
    html.Br(),
    html.Br(),
    html.Button(dcc.Link('projection avec l\'algorithme prophète', href='/Prophet')),
    html.Br(),
    html.Br(),
    html.Button(dcc.Link('projection avec l\'algorithme prophète les différentes composantes', href='/Prophete_components')),
    html.Br(),
    html.Br(),
    html.Button(dcc.Link('les metrics de prophète', href='/Prophet_metrics')),
    html.Br(),
    html.Br(),
    html.Button(dcc.Link('projection avec regression lineare', href='/regression_lineare')),
    html.Br(),
    html.Br(),
    html.Button(dcc.Link('projection avec SARIMA', href='/SARIMA'))
], style={'alignItems': 'center'})

liste_crypto = ['BTCBUSD', 'ETHBUSD','BNBUSDT', 'ALCXBTC', 'ALGOBTC', 'AAVEBTC',  'ADABTC', 'ADXBTC', 'AGIXBTC',  'ACHBTC', 'ACABTC'] #à remplacer par un appel API
frequency ='1d'

date_start = '2018-01-21T00:00:00.000Z'  # Replace with actual start date
#current_date = datetime.datetime.now()
#date_end = current_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
#date_end = '2023-09-21T00:00:00.000Z' # Replace with actual end date

current_utc_time = datetime.utcnow()

current_utc_time_with_timezone = current_utc_time.replace(tzinfo=pytz.UTC)
date_end = current_utc_time_with_timezone.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

layout_1 = html.Div([
    html.Div([
        html.H2('Evolution de la CryptoCurrency', style={'textAlign': 'center', 'color': 'mediumturquoise'}),
        html.Label("selectionner une Cryptocurrency:"),
        html.Div(dcc.Dropdown(
            id="dropdown",
            options=[{"label": CryptoCurrency, "value": CryptoCurrency} for CryptoCurrency in liste_crypto],
            value=None
        )),
        html.Label("Entrer en jour la profondeur souhaitée:"),
        dcc.Input(id="integer-input", type="number", value=None, step=1),
        html.Div(id="table-container"),
        dcc.Graph(id='price-plot')  
    ]),  # Première colonne
])

@app_binance.callback(
    [Output("table-container", "children"), Output('price-plot', 'figure')],
    [Input("dropdown", "value"), Input("integer-input", "value")]
)
def update_content(selected_CryptoCurrency, profondeur_saisie):
    
    table_content = f"vous avez sélectionné: {selected_CryptoCurrency}" if selected_CryptoCurrency else ""
    if profondeur_saisie is not None:
        table_content += f"<br>Le nombre de jour: {profondeur_saisie}"

    # Create and update the plot based on the selected dropdown value
    if selected_CryptoCurrency and profondeur_saisie:
        date_start2 = datetime.now() - timedelta(profondeur_saisie)
        date_end2  = datetime.now()
        base_url = "http://api_binance_alias:3000/get_price_short_information/"
        url = f"{base_url}?symbole={selected_CryptoCurrency}&frequency={frequency}&date_debut={date_start2}&date_fin={date_end2}"
        #print("000000000",url)
        with httpx.Client() as client:
            response =  client.get(url, timeout=20)
            data = response.json()

        fig, ax = plt.subplots(figsize=(12, 8))
        df_selected = pd.DataFrame(data)
        df_selected["close_time"] = df_selected["close_time"]
        df_selected["close"] = df_selected["close"].astype(float)
        
        df_selected.set_index('close_time').plot(ax=ax)
        ax.set_ylabel('end of day price')
        ax.set_xlabel('Date')
        plt.style.use('fivethirtyeight')

        # Convert the Matplotlib figure to Plotly-compatible format
        plotly_fig = go.Figure(data=go.Scatter(x=df_selected['close_time'], y=df_selected['close']))
        plotly_fig.update_layout(title_text="End of Day Price")
        
        return table_content, plotly_fig

    return table_content, {}

layout_2 = html.Div([
    html.H2('Prediction avec l\'algorithme prophète les différentes composantes', style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Label("Selectionner une Cryptocurrency:"),
    html.Div(dcc.Dropdown(
        id="dropdown_pro",
        options=[{"label": CryptoCurrency, "value": CryptoCurrency} for CryptoCurrency in liste_crypto],
        value=None
    )),
    html.Label("Entrer votre horizon de projection en jour:"),
    dcc.Input(id="integer-input_pro", type="number", value=None, step=1),
    html.Button("Rafraîchir", id="refresh-button"),
    html.Div([
        dcc.Loading(
            id="loading-forecast",
            type="default",
            children=[
                html.Div(id="table-container2"),
                #dcc.Graph(id='forecast-plot'),
                html.Div([
                    html.H2("Components Plot", style={'textAlign': 'center'}),
                    html.H3(id="selected-currency", style={'textAlign': 'center', 'color': 'mediumturquoise'}),
                    html.Img(id='components_img', src='assets/components_plot.png')
                ])
            ]
        )
    ])
])  # First column

@app_binance.callback(
    [Output("table-container2", "children"), Output('components_img', 'src'), Output("selected-currency", "children")],
    [Input("refresh-button", "n_clicks")],
    [State("dropdown_pro", "value"), State("integer-input_pro", "value")]
)
def generate_prophet_plots(n_clicks, selected_CryptoCurrency, prediction_saisie ):
    if n_clicks and selected_CryptoCurrency and prediction_saisie:
        
        table_content2 = f"vous avez sélectionné: {selected_CryptoCurrency}" 
        table_content2 += f"<br>Le nombre de jour: {prediction_saisie}"
        
        #date_start = '2019-01-21T00:00:00.000Z'  # Replace with actual start date
        #date_end = '2023-08-21T00:00:00.000Z' # Replace with actual end date
        
        base_url = "http://api_binance_alias:3000/get_price_short_information/" 
        url3 = f"{base_url}?symbole={selected_CryptoCurrency}&frequency={frequency}&date_debut={date_start}&date_fin={date_end}" 
    
        with httpx.Client() as client2:
            #response =  client2.get(url3)
            response =  client2.get(url3, timeout=20)
            data = response.json()  # Convertir la réponse JSON en données Python

        if isinstance(data, list):  # Si la réponse est une liste de dictionnaires
            df_selec = pd.DataFrame(data)
        elif isinstance(data, dict):  # Si la réponse est un dictionnaire
            df_selec = pd.DataFrame([data])
        
        df_selec = df_selec.iloc[:, -2:]

        
        df_selec = df_selec.rename(columns={'close_time': 'ds', 'close': 'y'})
        my_model = Prophet(interval_width=0.90)
        my_model.fit(df_selec)
        future_dates = my_model.make_future_dataframe(periods=prediction_saisie, freq='D')
        forecast2 = my_model.predict(future_dates)
        
        components_fig = my_model.plot_components(forecast2)
        table_content2 = f"vous avez sélectionné: {selected_CryptoCurrency}<br>Le nombre de jour: {prediction_saisie}"
          # Save the plot as an image file
        #forecast_plot.update_layout(title='Forecast with Uncertainty')
        temp_file_path = "assets/components_plot.png"
        components_fig.savefig(temp_file_path)
        #components_fig.savefig('assets/components_plot.png')
        
        return table_content2, temp_file_path, f"Currency choisie : {selected_CryptoCurrency}"
    elif (selected_CryptoCurrency is not NONE ):
        table_content2 =''
        temp_file_path = "assets/components_plot.png"
        return table_content2, temp_file_path ,''

#######################################################################################
layout_22 = html.Div([
    html.H2('Prediction avec l\'algorithme prophète', style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Label("Selectionner une Cryptocurrency:"),
    html.Div(dcc.Dropdown(
        id="dropdown_pro",
        options=[{"label": CryptoCurrency, "value": CryptoCurrency} for CryptoCurrency in liste_crypto],
        value=None
    )),
    html.Label("Entrer votre horizon de projection en jour:"),
    dcc.Input(id="integer-input_pro", type="number", value=None, step=1),
    html.Button("Rafraîchir", id="refresh-button"),
    html.Div([
        dcc.Loading(
            id="loading-forecast",
            type="default",
            children=[
                html.Div(id="table-container3"),
                dcc.Graph(id='forecast-plot')            
            ]
        )
    ])
])  # First column

@app_binance.callback(
    [Output("table-container3", "children"), Output('forecast-plot', 'figure')],
    [Input("refresh-button", "n_clicks")],
    [State("dropdown_pro", "value"), State("integer-input_pro", "value")]
)
def generate_prophet_plots(n_clicks, selected_CryptoCurrency, profondeur_saisie ):
    if n_clicks and selected_CryptoCurrency and profondeur_saisie:
       # print("je suis la ",selected_CryptoCurrency,profondeur_saisie,n_clicks)
        table_content3 = ''
        table_content3 = f"vous avez sélectionné: {selected_CryptoCurrency}" 
        table_content3 += f"<br> Le nombre de jour: {profondeur_saisie}"
        #date_start = '2019-01-21T00:00:00.000Z'  # Replace with actual start date
        #date_end = '2023-08-21T00:00:00.000Z' # Replace with actual end date
        
        base_url = "http://api_binance_alias:3000/Prophete_Prediction/"
        url2 = f"{base_url}?symbole={selected_CryptoCurrency}&frequency={frequency}&date_debut={date_start}&date_fin={date_end}&Nb_of_day_Prediction={profondeur_saisie}"
        
        with httpx.Client() as client:
            #response = client.get(url2)
            response =  client.get(url2, timeout=20)
            Data = response.json()
        
        forecast = pd.DataFrame(Data)
        
        forecast_plot = px.line(forecast, x='ds', y='yhat', title=f'Forecast with Uncertainty for {selected_CryptoCurrency} for {profondeur_saisie} days',)

        return table_content3, forecast_plot
    else:
        table_content3 =''
        return table_content3, {}

###################################################################################

layout_6 = html.Div([
    html.H2('Fiabilité de l\'algorithme prophète', style={'textAlign': 'center', 'color': 'mediumturquoise'}),
    html.Label("Selectionner une Cryptocurrency:"),
    html.Div(dcc.Dropdown(
        id="dropdown_pro",
        options=[{"label": CryptoCurrency, "value": CryptoCurrency} for CryptoCurrency in liste_crypto],
        value=None
    )),
    html.Label("Entrer la durée de test en jour:"),
    dcc.Input(id="integer-input_pro", type="number", value=None, step=1),
    html.Button("Rafraîchir", id="refresh-button"),
    html.Div(id="table-container6"),
    dcc.Graph(id='graph')   
])

@app_binance.callback(
    [Output("table-container6", "children"), Output('graph', 'figure')],
    #Output('graph', 'figure'),
    [Input("refresh-button", "n_clicks")],
    [State("dropdown_pro", "value"), State("integer-input_pro", "value")]
)
def generate_prophet_plots_metrics(n_clicks, selected_CryptoCurrency, profondeur_saisie):
    table_content6 = ''
    if n_clicks and selected_CryptoCurrency and profondeur_saisie:
        table_content6 = f"vous avez sélectionné: {selected_CryptoCurrency}" 
        table_content6 += f"<br>Le nombre de jour: {profondeur_saisie}"
        # Appel à l API
        #date_start = '2019-01-21T00:00:00.000Z'  # Replace with actual start date
        #date_end = '2023-08-21T00:00:00.000Z' # Replace with actual end date
        base_url = "http://api_binance_alias:3000/Prophete_Prediction_metrics/"
        url6 = f"{base_url}?symbole={selected_CryptoCurrency}&frequency={frequency}&date_debut={date_start}&date_fin={date_end}&Nb_of_day_Prediction={profondeur_saisie}"
        #print (url6)
        with httpx.Client() as client6:
            #response = client6.get(url6)
            response =  client6.get(url6, timeout=20)
            result6 = response.json()
        
        mae = result6["mae"]
        y_true = result6["y_true"]
        y_pred = result6["y_pred"]
        y_pousse = result6["y_pousse"]
        ds_values = y_pousse[profondeur_saisie:]
        # Création de graphiques ou d'autres éléments HTML pour afficher les résultats


        #mae_text = html.P(f"Mean Absolute Error: {mae}")
        #table_content6 += mae_text
        
        pred_trace = go.Bar(x=ds_values,y=y_pred, name='Predicted')
        true_trace = go.Bar(x=ds_values,y=y_true, name='True')

        graph_layout = go.Layout(title=f"Mean Absolute Error: {mae} , 'Predicted vs True Values'")
        #graph_layout= go.axvline(x= datetime.date(2023,8,23), color='red')
        #fig2 = go.Figure(data=[pred_trace, true_trace], layout=graph_layout)
        #table_content6 += f"Mean Absolute Error: {mae}"

# Create figure
        fig = go.Figure(data=[pred_trace, true_trace], layout=graph_layout)

        
        return table_content6, fig
    else:
        return table_content6,{}

################################################################################

layout_4 = html.Div([
    html.Div([
        html.H2('Regression lineare & métrics ', style={'textAlign': 'center', 'color': 'mediumturquoise'}),
        html.Label("selectionner une Cryptocurrency:"),
        html.Div(dcc.Dropdown(
            id="dropdown",
            options=[{"label": CryptoCurrency, "value": CryptoCurrency} for CryptoCurrency in liste_crypto],
            value=None
        )),
        html.Label("Entrer le ratio du jeu de test:"),
        dcc.Input(id="integer-input", type="number", value=None, step=0.01 , min =0, max=1),
        html.Button("Rafraîchir", id="refresh-button"),
        html.Div(id="table-container4"),
        dcc.Graph(id='regression_graph')
    ]),  
])

@app_binance.callback(
    [Output("table-container4", "children"), Output('regression_graph', 'figure')],
    [Input("dropdown", "value"), Input("integer-input", "value"), Input("refresh-button", "n_clicks")]
)
def update_content(selected_CryptoCurrency, ratio, n_clicks):
    table_content4 = ""
    figure_com = {}
    # Create and update the plot based on the selected dropdown value
    if n_clicks and selected_CryptoCurrency and ratio   :
        table_content4 = f"vous avez sélectionné: {selected_CryptoCurrency}" if selected_CryptoCurrency else ""
        table_content4 += f"<br> Le ratio choisi : {ratio}"
        base_url = "http://api_binance_alias:3000/regression_Lineare/"
        url = f"{base_url}?symbole={selected_CryptoCurrency}&frequency={frequency}&date_debut={date_start}&date_fin={date_end}&size_test={ratio}"
        
        with httpx.Client() as client:
            response =  client.get(url)
            data = response.json()
       
        score_train = data["score_train"]
        Score_test = data["Score_test"]
        y_true = data["y_true"]
        y_pred = data["prediction_test"]
        pred_tomorrow= data["pred_tomorrow"]
        
        Len = len(y_pred)
        number_list = [i for i in range(1, Len+1)]
    
        #print(number_list)
        pred_trace = go.Bar(x=number_list, y=y_pred, name='Predicted')
        true_trace = go.Bar(x=number_list , y=y_true, name='True')
        #print ('FINIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
        #next_day = date_end + timedelta(days=1)
        graph_layout = go.Layout(title={'text': f"Predicted vs True Values \n Score train: {score_train}, Score test: {Score_test} Prédiction du prix pour demain {date_end} est : {pred_tomorrow} ",
                            'font': {'color': 'green'}}
                            )
# Create figure
        figure_com = go.Figure(data=[pred_trace, true_trace], layout=graph_layout)
       
        return table_content4 , figure_com
    table_content4=""
    return table_content4, {}

###################################################################################
@app_binance.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/cryptocurrency':
        return layout_1
    if pathname == '/Prophete_components':
        return layout_2
    if pathname == '/Prophet':
        return layout_22
    elif pathname == '/Prophet_metrics':
        return layout_6
    elif pathname == '/regression_lineare':
        return layout_4
    else:
        return index_page

if __name__ == '__main__':
    app_binance.run_server(debug=True, host="0.0.0.0", port=2000)
