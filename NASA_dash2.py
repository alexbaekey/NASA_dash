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

#TODO black background, white text, ariel
#Make into more of "Regional what-if analysis tool" with dropdowns for scenarios
#Clean!!
# layout including sidebar and central plots, side plots on right

# coastal heights
h_baltimore = 3.0
h_somewhere = 7.0

csv_url = 's3://nasa-dash-blue-marble/8575512_meantrend.csv'
#img_url = 'https://i.ibb.co/0jTLg38/maryland-coast.jpg'
img_url = 'https://i.ibb.co/58169Z1/image001.png'

df_anna_sea = pd.read_csv(csv_url, index_col=False)

fig2 = go.Figure() # or any Plotly Express function e.g. px.bar(...)
fig3 = go.Figure()

x1=np.random.randn(500)
fig4 = go.Figure(data=[go.Histogram(x=x1)])

y_0 = np.random.randn(75)-0.5
y_1 = np.random.randn(75)+0.5
y_2 = np.random.randn(120)-0.4

fig5 = go.Figure()

fig5.add_trace(go.Box(y=y_0))
fig5.add_trace(go.Box(y=y_1))
fig5.add_trace(go.Box(y=y_2))

#maybe this next? https://plotly.com/python/scattermapbox/


fig2.add_layout_image(
        dict(    
            source=img_url,
            xref="x",
            yref="y",
            x=0,
            y=4,
            sizex=6,
            sizey=4,
            sizing="stretch",
            opacity=0.95,
            layer="below",
            )
        )

fig3.update_xaxes(title="Year")
fig3.update_yaxes(title="Mean Sea Level (MSL)")

fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', range=[0,6],tickvals=[0,1,2,3,4,5])
fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink', range=[0,4],tickvals=[0,1,2,3])


df_anna_sea['Year']=df_anna_sea['Year']+df_anna_sea['Month']/12


length=len(df_anna_sea['Year'])
x = df_anna_sea['Year'].values.reshape(length,1)
y = df_anna_sea['Monthly_MSL'].values.reshape(length,1)
regr = linear_model.LinearRegression()
regr.fit(x,y)

#for trace #4 in fig (fill area)
#TODO replace random values with some sort of anomaly range, ...or something

x_fill = np.arange(1930,3000,10) 
y_up   = np.zeros(len(x_fill))
y_low  = np.zeros(len(x_fill))

for ind, i in enumerate(x_fill):
    y_up[ind]   = float(regr.coef_)*i+float(regr.intercept_)+(2*np.random.sample())
    y_low[ind]  = float(regr.coef_)*i+float(regr.intercept_)-(2*np.random.sample())



#NEW WAY https://community.plotly.com/t/multiple-traces-with-a-single-slider-in-plotly/16356/2

trace_list_1 = []
trace_list_2 = []
trace_list_3 = []
trace_list_4 = []
trace_list_5 = []

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
        y=[h_baltimore,h_somewhere],
        mode="markers+text",
        text=["h_baltimore","h_somewhere"],
        textposition="top right",
        showlegend=False
    )
    trace_list_3.append(temp3)

for step in np.arange(1930,3000,10):
    temp4=go.Scatter(
        x=x_fill,
        y=y_low,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    )
    trace_list_4.append(temp4)

for step in np.arange(1930,3000,10):
    temp5=go.Scatter(
        x=x_fill,
        y=y_up,
        marker=dict(color="#444"),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(68,68,68,0.3)',
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    )
    trace_list_5.append(temp5)

fig = go.Figure(
        data=trace_list_1+trace_list_2+trace_list_3+trace_list_4+trace_list_5,
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
    tickcolor='black',
    font=dict(
        color='black'
        )
)]

fig.update_layout(
    sliders=sliders,
    title = "Sea-Level Prediciton in Maryland",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel',
    shapes=[
        dict(
            type='line',
            yref='y', y0= h_baltimore, y1= h_baltimore,
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
    height=400,
    title= "Baltimore demographics",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )

fig3.update_layout(
    width = 500,
    height = 500,
    title = "Historical MSL data - Baltimore - NOAA",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )

fig4.update_layout(
    width = 500,
    height = 500,
    title = "Bar Charts!",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )

fig5.update_layout(
    width = 500,
    height = 500,
    title = "Box Plots!",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )







#app layout stuff
img2_url = 'https://i.ibb.co/12p7vTF/bluemarbleheader.png' 


app = dash.Dash()
server=app.server
app.layout = html.Div(children=[ 
    html.Div(
        '''
        An example of a climate dashboard
        '''
    ),
    html.Img(
        src=img2_url,
        style={'width':'100%'}
    ),
    #divide up into two big divs, sidebar on left, graphs on right?
    html.Div(
        dcc.Graph(figure=fig),
    ),
    html.Div(
        dcc.Graph(figure=fig3),
        style={'width': '48%', 'align': 'right', 'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(figure=fig2),
        style={'width': '48%','display': 'inline-block','backgroundColor':'black'}
    ),
    html.Div(
        dcc.Graph(figure=fig4),
        style={'width': '48%','display': 'inline-block','backgroundColor':'black'}
    ),
    html.Div(
        dcc.Graph(figure=fig5),
        style={'width': '48%','display': 'inline-block','backgroundColor':'black', 'align':'right'}
    ),
    ], style={'backgroundColor':'black'}
)
if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)  # Turn off reloader if inside Jupyter nb or deploying in Heroku



