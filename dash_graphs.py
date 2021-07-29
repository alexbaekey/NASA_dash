import dash
import dash_core_components as dcc
import dash_html_components as html

app=dash.Dash()

#starts a div tag
app.layout = html.Div('Dash tutorials') 

if __name__ == '__main__':
    app.run_server(debug=True)
