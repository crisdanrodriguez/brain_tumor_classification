import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div(children = [
    html.Font('Brain Tumor Classification', className = 'main_title'),
    html.Br(),
    html.Font('Cristian Rodriguez', className = 'main_subtitle'),
    html.Br(),
    html.Div([
        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '70%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': 'auto',
            'margin-top': '10px',
            'margin-bottom': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),

    ], className = 'second_div')

], className = 'main_div')

if __name__ == '__main__':
    app.run_server()