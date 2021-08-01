#https://plotly.com/python/sliders/

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

#load up csv from NOAA, not working ...
#url = "http://tidesandcurrents.noaa.gov/sltrends/data/8575512_meantrend.csv"
#response = requests.post(url,data={'':})
#df_anna_sea = pd.read_csv(io.StringIO(response.decode('utf-8')))

df_anna_sea = pd.read_csv('8575512_meantrend.csv', index_col=False)

fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)
fig2 = go.Figure() # or any Plotly Express function e.g. px.bar(...)
fig3 = go.Figure()


img = Image.open("maryland_coast.jpg")

fig2.add_layout_image(
        dict(    
            source=img,
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
fig3.update_yaxes(title="Historical MSL")

fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', range=[0,4],tickvals=[0,1,2,3])
fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', range=[0,4],tickvals=[0,1,2,3])

fig.update_xaxes(title="Year")
fig.update_yaxes(title="trend")


df_anna_sea['Year']=df_anna_sea['Year']+df_anna_sea['Month']/12

#fig.add_trace(
#        go.Scatter(
#            x=df_anna_sea['Year'],
#            y=df_anna_sea['Monthly_MSL'],
#            mode='markers'
#            )
#        )

length=len(df_anna_sea['Year'])
x = df_anna_sea['Year'].values.reshape(length,1)
y = df_anna_sea['Monthly_MSL'].values.reshape(length,1)
regr = linear_model.LinearRegression()
regr.fit(x,y)


# add traces per step
for step in np.arange(1930, 3000, 1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="year = " + str(1930+step),
            x=np.arange(1930, step, 1),
            #y=np.sin(2.22*np.arange(2019,2500,1)+1)))
            #y=0.00371*np.arange(2019,3000,1)-7.382 # 3.71mm/year mean trend, -7.382m at year 0
            # using values from tidesandcurrents.noaa.gov for Annapolis, MD 
            y=float(regr.coef_)*np.arange(1930, 3000, 1)+float(regr.intercept_),
        )
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
        mode='markers'
    )
)

# display inital trace value on timeline
fig.data[1].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Projection for year: " + str(1930+i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=70,
    currentvalue={"prefix": "Year: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    title = "Sea-Level Prediciton in Maryland"
    )

fig2.update_layout(
    width=400,
    height=400,
    title= "Battleships"
    )

fig3.update_layout(
    width = 600,
    height = 400,
    title = "trendline"
    )

app = dash.Dash()
app.layout = html.Div([
    html.Div(
        dcc.Graph(figure=fig2),
        style={'width': '48%','display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(figure=fig3),
        style={'width': '48%', 'align': 'right', 'display': 'inline-block'}
    ),

    html.Div(
        dcc.Graph(figure=fig)
    ),
    ]
    #style={'columnCount': 2}
    #style={'width': '48%','display': 'inline-block'}
)


# known to work:
#app.layout = html.Div([ # dash.plotly.com/layout
#    dcc.Graph(figure=fig)
#    dcc.Graph(figure=fig2),
#    dcc.Graph(figure=fig3),
#    ],
#    #style={'columnCount': 2}
#    #style={'width': '48%','display': 'inline-block'}
#)







app.run_server(debug=True, use_reloader=True)  # Turn off reloader if inside Jupyter
