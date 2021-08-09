import plotly.graph_objects as go # or plotly.express as px
#import plotly.tools as ptt
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
#from scipy.interpolate import griddata
#import matplotlib.pyplot as plt

##### data downloads, csv reading #####

csv_url      = 's3://nasa-dash-blue-marble/8575512_meantrend.csv'
#img_url     = 'https://i.ibb.co/0jTLg38/maryland-coast.jpg'
img_url      = 'https://i.ibb.co/58169Z1/image001.png'
#img_url      = 'https://i.ibb.co/RShLgD5/image001.png'
#obtained from https://topex.ucsd.edu/cgi-bin/get_data.cgi
#balt_alt_url = 's3://nasa-dash-blue-marble/balt_alts.csv'

#df_balt_alt = pd.read_csv(balt_alt_url,index_col=False)
df_balt_sea = pd.read_csv(csv_url, index_col=False)

#balts = df_balt_alt.to_numpy()

##### static plots #####

fig2 = go.Figure()
fig3 = go.Figure()
fig6 = go.Figure()

x1=np.random.randn(500)
fig4 = go.Figure(data=[go.Histogram(x=x1)])

y_0 = np.random.randn(75)-0.5
y_1 = np.random.randn(75)+0.5
y_2 = np.random.randn(120)-0.4

fig5 = go.Figure()

fig5.add_trace(go.Box(y=y_0,showlegend=False))
fig5.add_trace(go.Box(y=y_1,showlegend=False))
fig5.add_trace(go.Box(y=y_2,showlegend=False))

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

fig2.update_xaxes(showgrid=False, gridwidth=1, gridcolor='LightPink', range=[0,6],tickvals=[0,1,2,3,4,5])
fig2.update_yaxes(showgrid=False, gridwidth=1, gridcolor='LightPink', range=[0,4],tickvals=[0,1,2,3])

df_balt_sea['Year']=df_balt_sea['Year']+df_balt_sea['Month']/12

length=len(df_balt_sea['Year'])
x = df_balt_sea['Year'].values.reshape(length,1)
y = df_balt_sea['Monthly_MSL'].values.reshape(length,1)
regr = linear_model.LinearRegression()
regr.fit(x,y)

#for trace #4 in fig (fill area)
#TODO replace random values with some sort of anomaly range, ...or something

x_fill = np.arange(1930,3000,10) 
y_up   = np.zeros(len(x_fill))
y_low  = np.zeros(len(x_fill))
y_up1   = np.zeros(len(x_fill))
y_low1  = np.zeros(len(x_fill))
y_up2   = np.zeros(len(x_fill))
y_low2  = np.zeros(len(x_fill))



for ind, i in enumerate(x_fill):
    y_up[ind]   = float(regr.coef_)*i+float(regr.intercept_)+(2*np.random.sample())
    y_low[ind]  = float(regr.coef_)*i+float(regr.intercept_)-(2*np.random.sample())
    y_up1[ind]   = 1.5*float(regr.coef_)*i+float(regr.intercept_)+(2*np.random.sample())
    y_low1[ind]  = 1.5*float(regr.coef_)*i+float(regr.intercept_)-(2*np.random.sample())
    y_up2[ind]   = (0.5*float(regr.coef_)*i)**2+float(regr.intercept_)+(2*np.random.sample())
    y_low2[ind]  = (0.5*float(regr.coef_)*i)**2+float(regr.intercept_)-(2*np.random.sample())



fig3.add_trace(
    go.Scatter(
        x=df_balt_sea['Year'],
        y=df_balt_sea['Monthly_MSL'],
        mode='markers',
        name="Historical MSL",
        showlegend=False
    )
)

fig3.add_trace(
    go.Scatter(
        x=df_balt_sea['Year'],
        y=float(regr.coef_)*df_balt_sea['Year']+float(regr.intercept_),
        line = dict(color='firebrick', width=4),
        name = "trendline",
        showlegend=False
        )
    )


#Long = balts[:,0] 
#Lat  = balts[:,1] 
#Elev = balts[:,2]
#pts = 100000
#
#t1 = np.linspace(np.min(Long),np.max(Long),int(np.sqrt(pts)))
#t2 = np.linspace(np.min(Lat),np.max(Lat),int(np.sqrt(pts)))
#[x,y]=np.meshgrid(t1,t2)
#
#z=griddata((Long,Lat),Elev, (x,y), method='linear')
#x=np.matrix.flatten(x)
#y=np.matrix.flatten(y)
#z=np.matrix.flatten(z)

#altfig = plt.figure()
##altfig.patch.set_facecolor('black')
#plt.xlabel('Longitude (degrees)')
#plt.ylabel('Latitude (degrees)')
#plt.scatter(x,y,1,z,cmap='terrain')
##plt.colorbar(label='elevation above sea level (m)')
#ax = plt.gca()
#ax.set_aspect('equal')
#fig6 = ptt.mpl_to_plotly(altfig)


##### Dynamic plot #####

#NEW WAY https://community.plotly.com/t/multiple-traces-with-a-single-slider-in-plotly/16356/2

trace_list_1  = []
trace_list_2  = []
trace_list_4  = []
trace_list_5  = []
trace_list_1a = []
trace_list_4a = []
trace_list_5a = []
trace_list_1b = []
trace_list_4b = []
trace_list_5b = []

for step in np.arange(1930,3000,10):
    temp=go.Scatter(
        x=np.arange(1930, 3000, 10),
        y=float(regr.coef_)*np.arange(1930, 3000, 10)+float(regr.intercept_),
        line = dict(color='blue', width=4),
        name="Best-case",
        showlegend=False
        #legendgroup="Best-case"
        )                   
    trace_list_1.append(temp)

for step in np.arange(1930,3000,10):
    temp2=go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=3),
            name="year = " + str(step),
            x=[step,step],
            y=[0,30],
            showlegend=False
        )                   
    trace_list_2.append(temp2)

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

for step in np.arange(1930,3000,10):
    tempa=go.Scatter(
        x=np.arange(1930, 3000, 10),
        y=1.5*float(regr.coef_)*np.arange(1930, 3000, 10)+float(regr.intercept_),
        line = dict(color='green', width=4),
        showlegend=False,
        name="Average-case"
        #legendgroup="Average-case"
        )                   
    trace_list_1a.append(tempa)

for step in np.arange(1930,3000,10):
    temp4a=go.Scatter(
        x=x_fill,
        y=y_low1,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    )
    trace_list_4a.append(temp4a)

for step in np.arange(1930,3000,10):
    temp5a=go.Scatter(
        x=x_fill,
        y=y_up1,
        marker=dict(color="#444"),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(68,68,68,0.3)',
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    )
    trace_list_5a.append(temp5a)

for step in np.arange(1930,3000,10):
    tempb=go.Scatter(
        x=np.arange(1930, 3000, 10),
        y=(0.5*float(regr.coef_)*np.arange(1930, 3000, 10))**2+float(regr.intercept_),
        line = dict(color='firebrick', width=4),
        showlegend=False,
        name="Worst-case"
        #legendgroup="Worst-case"
        )                   
    trace_list_1b.append(tempb)

for step in np.arange(1930,3000,10):
    temp4b=go.Scatter(
        x=x_fill,
        y=y_low2,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    )
    trace_list_4b.append(temp4b)

for step in np.arange(1930,3000,10):
    temp5b=go.Scatter(
        x=x_fill,
        y=y_up2,
        marker=dict(color="#444"),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(68,68,68,0.3)',
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False
    )
    trace_list_5b.append(temp5b)



#assemble traces
fig = go.Figure(
        data=trace_list_1+trace_list_2+trace_list_4+trace_list_5+trace_list_1a+trace_list_4a+trace_list_5a+trace_list_1b+trace_list_4b+trace_list_5b,
        layout_xaxis_range=[1930,2980], #magic underscores used! Look up if confused
        layout_yaxis_range=[0,30],
        layout_xaxis_title="Year",
        layout_yaxis_title="Height (m)",
        )

# display inital trace value on timeline
#fig.data[5].visible = True

# Create and add slider
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
    active=0,
    currentvalue={"prefix": "Year: "},
    #pad={"t": 50},
    steps=steps,
    tickcolor='black',
    font=dict(
        color='black'
        )
)]



##### Figure layouts #####

fig.update_layout(
    sliders=sliders,
    title = "Sea-Level Prediciton in Baltimore, Maryland",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel',
    )

fig2.update_layout(
    height=500,
    title= "Baltimore demographics",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )

fig3.update_layout(
    height = 400,
    title = "Historical MSL data - Baltimore - NOAA",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )

fig4.update_layout(
    height = 300,
    title = "Bar Charts!",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )

fig5.update_layout(
    height = 300,
    title = "Box Plots!",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )

fig6.update_layout(
    title = "Altitude from Sea-Level",
    paper_bgcolor='rgb(0,0,0)',
    font_color='white',
    font_family='ariel'
    )



#app layout 
img2_url = 'https://i.ibb.co/12p7vTF/bluemarbleheader.png' 
img3_url = 'https://i.ibb.co/DgTTKCs/fake-sidebar.png'
img4_url = 'https://i.ibb.co/Ytsvqj3/bluemarble-logo.png' 
img5_url = 'https://www.climate.gov/sites/default/files/sealevel_contributors_graph_SOTC2018_620.jpg'



##### app layout #####

app = dash.Dash()
server=app.server
app.layout = html.Div(children=[ 
    html.Img(
        src=img2_url,
        style={'width':'100%','margin-bottom':'40px'}
    ),
    #divide up into two big divs, sidebar on left, graphs on right?
    html.Img(
        src=img3_url,
        style={'width':'13%','align':'left','display':'inline-block'}
    ),
    html.Div(
        children=[
            html.H1(
                'Regional Decision Support - Baltimore, MD',
                style={ 'color':'white',
                        'text-align':'center',
                        'font_family':'ariel'}
            ),
            #html.H1(
            #    'In Baltimore, MD, flooding is the most prominant climate risk.',
            #    style={ 'color':'white',
            #            'text-align':'left',
            #            'font_family':'ariel',
            #            'font-size':'20px',
            #            'padding-left':'40px'}
            #),
            html.H1(
                'Demographics Map',
                style={
                    'font_color':'white',
                    'font_family':'ariel',
                    'font-size':'18px'
                }
            ),
            html.Img(
                src=img_url,
                title='Demographics map',
                style={
                    'padding-left':'100px',
                    'padding-right':'100px',
                    }
            ),
            html.H2(
                '\"What-if\" Prediction',
                style={ 'color':'white',
                        'text-align':'left',
                        'margin-top':'50px',
                        'font_family':'ariel',
                        'padding-left':'40px'}
            ), 
            html.Div(
                dcc.Graph(figure=fig),
            ),
            #html.Div(
            #    dcc.Graph(figure=fig6),
            #    style={'backgroundColor':'black'}
            #)
        ],
        style={ 'width':'50%',
                'vertical-align':'top',
                'display':'inline-block',
                'margin-bottom':'30px',
            }
    ),
    html.Div(
        children=[   
        html.H2(
            'For demostration purposes only... Some data taken from NOAA, some randomized.',
            style={ 'width':'45%',
                    'color':'white',
                    'align':'left',
                    'display':'inline-block',
                    'font_family':'ariel',
                    'font-size':'15px',
                    'padding-left':'10px',
                    'padding-right':'10px'}
        ),
        html.Img(
            src=img4_url,
            style={'width':'50%',
            'margin-bottom':'15px',
            'vertical-align':'top',
            'align':'right',
            'display':'inline-block'
            }
        ), 
        html.Img(
            src=img5_url,
            style={'width':'93%',
            'padding-left':'20px',
            'padding-right':'20px'
            }
        ),
        html.Div(
            dcc.Graph(figure=fig3),
            style={
                'width':'98%'}
            ),
        html.Div(children=[
            html.Div(
                dcc.Graph(figure=fig4),
                style={'width': '48%','display':'inline-block','align':'left'}
            ),
            html.Div(
                dcc.Graph(figure=fig5),
                style={'width': '48%','display':'inline-block','align':'right'}
            )
            ],
            style={
                'width':'98%'
            }
        )    
        ],
        style={ 'width':'35%',
                'vertical-align':'top',
                'align':'right',
                'display':'inline-block',
                'border':'3px solid white'
                }
        )
    ], 
    style={'backgroundColor':'black'}
)



##### run app #####

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)  # Turn off reloader if inside Jupyter nb or deploying in Heroku



