import plotly.graph_objects as go # or plotly.express as px
from PIL import Image
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import os
import pandas as pd
import requests
import io
from sklearn import linear_model


# GRID POSITIONS
# (1.5,0.5) = Cove Point
# (2.5,1.5) = Cambridge
# (1.5,2.5) = Annapolis

# coastal heights
h_annapolis = 3.0
h_somewhere = 7.0

csv_url = 's3://nasa-dash-blue-marble/8575512_meantrend.csv'
#img_url = 's3://nasa-dash-blue-marble/maryland_coast.jpg'
img_url = 'https://i.ibb.co/0jTLg38/maryland-coast.jpg'
# This image link works! Was having so much trouble reading from S3 to image
# however, website will take down link in a month
# img = Image.open(img_url)

df_anna_sea = pd.read_csv(csv_url, index_col=False)

fig2 = go.Figure() # or any Plotly Express function e.g. px.bar(...)
fig3 = go.Figure()


fig2.add_layout_image(
        dict(    
            source=img_url,
            xref="x",
            yref="y",
            x=0,
            y=4,
            sizex=4,
            sizey=4,
            sizing="stretch",
            opacity=0.5,
            layer="below",
            )
        )

fig3.update_xaxes(title="Year")
fig3.update_yaxes(title="MSL")

fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', range=[0,4],tickvals=[0,1,2,3])
fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', range=[0,4],tickvals=[0,1,2,3])


df_anna_sea['Year']=df_anna_sea['Year']+df_anna_sea['Month']/12


length=len(df_anna_sea['Year'])
x = df_anna_sea['Year'].values.reshape(length,1)
y = df_anna_sea['Monthly_MSL'].values.reshape(length,1)
regr = linear_model.LinearRegression()
regr.fit(x,y)

 

#NEW WAY https://community.plotly.com/t/multiple-traces-with-a-single-slider-in-plotly/16356/2

trace_list_1 = []
trace_list_2 = []
trace_list_3 = []

for step in np.arange(1930,3000,10):
    temp=go.Scatter(
        x=np.arange(1930, 3000, 10),
        y=float(regr.coef_)*np.arange(1930, 3000, 10)+float(regr.intercept_),
        line = dict(color='firebrick', width=4),
        showlegend=False
        )                   
    trace_list_1.append(temp)

for step in np.arange(1930,3000,10):
    temp2=go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=3),
            name="year = " + str(step),
            x=[step,step],
            y=[0,10],
            showlegend=False
        )                   
    trace_list_2.append(temp2)

for step in np.arange(1930,3000,10):
    temp3=go.Scatter(
        x=[1930,1930],
        y=[h_annapolis,h_somewhere],
        mode="markers+text",
        text=["h_annapolis","h_somewhere"],
        textposition="top right",
        showlegend=False
    )
    trace_list_3.append(temp3)

fig = go.Figure(
        data=trace_list_1+trace_list_2+trace_list_3,
        layout_xaxis_range=[1930,2980], #magic underscores used! Look up if confused
        layout_yaxis_range=[0,10],
        layout_xaxis_title="Year",
        layout_yaxis_title="Height (m)",
        )
    
    # Update grid
    #    for i in stations:    
    #        if(0.00371*step-7.382<stations['height']):
    #            fig2.add_shape
    #)

fig3.add_trace(
    go.Scatter(
        x=df_anna_sea['Year'],
        y=df_anna_sea['Monthly_MSL'],
        mode='markers',
        name="Historical MSL",
    )
)

fig3.add_trace(
    go.Scatter(
        x=df_anna_sea['Year'],
        y=float(regr.coef_)*df_anna_sea['Year']+float(regr.intercept_),
        line = dict(color='firebrick', width=4),
        name = "trendline"
        )
    )

# display inital trace value on timeline
fig.data[1].visible = True

# Create and add slider
#got 21 from (3030-1930)/50
steps = []
for i in range(107):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * 107},
            #{"visible": [False]*i},
              {"title": "Projection for year: " + str(1930+(i)*10)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    #active=1,
    currentvalue={"prefix": "Year: "},
    #pad={"t": 50},
    steps=steps,
    tickcolor='white',
    font=dict(
        color='white'
        )
)]

fig.update_layout(
    sliders=sliders,
    title = "Sea-Level Prediciton in Maryland",
    shapes=[
        dict(
            type='line',
            yref='y', y0= h_annapolis, y1=h_annapolis,
            xref='paper', x0=0, x1=1,

        ),
        dict(
            type='line',
            yref='y', y0= h_somewhere, y1=h_somewhere,
            xref='paper', x0=0, x1=1
        )
    ],
    )

fig2.update_layout(
    width=500,
    height=500,
    title= "Underwater?"
    )

fig3.update_layout(
    width = 600,
    height = 500,
    title = "Historical MSL data - Annapolis - NOAA"
    )

app = dash.Dash()
server=app.server
app.layout = html.Div([ 
    html.Div(
        dcc.Graph(figure=fig)
    ),
    html.Div(
        dcc.Graph(figure=fig3),
        style={'width': '48%', 'align': 'right', 'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(figure=fig2),
        style={'width': '48%','display': 'inline-block'}
    ),
    ]
)
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter nb or deploying in Heroku



