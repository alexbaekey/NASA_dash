import dash
import dash_core_components as dcc
import dash_html_components as html

app=dash.Dash()

#starts a div tag
app.layout = html.Div(children=[
    html.H1('Dash tutorials'),
    dcc.Graph(id='example',
        figure={
            'data': [
                {'x':[1,2,3,4,5,6,7], 'y':[100,101,104,107,91,108,101], 'type':'line', 'name':'boats'},
                {'x':[1,2,3,4,5,6,7], 'y':[100,101,104,107,91,108,101], 'type':'bar', 'name':'cars'},
                ],
            'layout': {
                'title':'Basic Dash Example'
                }
            })

    ]) 

if __name__ == '__main__':
    app.run_server(debug=True)




