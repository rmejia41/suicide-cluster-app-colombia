#Complete map with marker clusters
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
import folium
from folium.plugins import MarkerCluster
from dotenv import load_dotenv


# Set Matplotlib backend to 'Agg' to avoid GUI-related issues
plt.switch_backend('Agg')

# Load environment variables from .env file
load_dotenv()

# Load data
file_path = 'C:/Users/Biu9/OneDrive - CDC/Python files/Colombia/Data/Suicidios_Colombia_2016_2019_merged.xlsx'
data = pd.read_excel(file_path)

# Correct mismatched data based on DEPARTAMENTO and MUNICIPIO
def correct_coordinates(data):
    corrections = {
        ('Antioquia', 'Medellin'): (6.25184, -75.56359),
        # Add more corrections here
    }

    for (dept, mun), (lat, lon) in corrections.items():
        data.loc[(data['DEPARTAMENTO'] == dept) & (data['MUNICIPIO'] == mun), ['LATITUD', 'LONGITUD']] = lat, lon

    return data

data = correct_coordinates(data)

# Rename columns for English labels
data = data.rename(columns={
    'Año del hecho': 'Year',
    'DEPARTAMENTO': 'Department'
})

# Function to create Seaborn line plots
def create_seaborn_plot(plot_type, data):
    plt.figure(figsize=(8, 5))  # Adjust the size here
    if plot_type == 'year':
        data_grouped = data.groupby('Year').size().reset_index(name='counts')
        sns.lineplot(data=data_grouped, x='Year', y='counts', marker='o')
        plt.xticks(data_grouped['Year'])
        plt.title('Trend in Suicides by Year', loc='right')
    elif plot_type == 'year_gender':
        data_grouped = data.groupby(['Year', 'Sexo de la victima']).size().reset_index(name='counts')
        sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Sexo de la victima', marker='o', palette='Set1')
        plt.xticks(data_grouped['Year'].unique())
        plt.title('Trend in Suicides by Year and Gender', loc='right')
    elif plot_type == 'year_age':
        data_grouped = data.groupby(['Year', 'Grupo de edad de la victima']).size().reset_index(name='counts')
        sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Grupo de edad de la victima', marker='o', palette='tab20')
        plt.xticks(data_grouped['Year'].unique())
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Trend in Suicides by Year and Age Group', loc='right')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# Function to generate Folium map with dynamic markers
def generate_map(filtered_data, initial=False):
    folium_map = folium.Map(location=[4.570868, -74.297333], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(folium_map)
    color_map = {
        'ANTIOQUIA': '#1f77b4',
        'NORTE DE SANTANDER': '#ff7f0e',
        'META': '#2ca02c',
        'CHOCO': '#d62728',
        'HUILA': '#9467bd',
        'BOLIVAR': '#8c564b',
        'CUNDINAMARCA': '#e377c2',
        'CESAR': '#7f7f7f',
        'SANTANDER': '#bcbd22',
        'CALDAS': '#17becf',
        'CASANARE': '#aec7e8',
        'LA GUAJIRA': '#ffbb78',
        'VALLE DEL CAUCA': '#98df8a',
        'NARINO': '#ff9896',
        'MAGDALENA': '#c5b0d5',
        'BOYACA': '#c49c94',
        'TOLIMA': '#f7b6d2',
        'RISARALDA': '#c7c7c7',
        'ARAUCA': '#dbdb8d',
        'CAUCA': '#9edae5',
        'QUINDIO': '#ff9da7',
        'ATLANTICO': '#b6d7a8',
        'CAQUETA': '#d4a5a5',
        'BOGOTA D.C.': '#ffcc00',
        'CORDOBA': '#999999',
        'SUCRE': '#ff6666',
        'GUAVIARE': '#cc99ff',
        'VAUPES': '#99cc00',
        'VICHADA': '#ffccff',
        'GUAINIA': '#ff9933',
        'AMAZONAS': '#66cccc',
        'PUTUMAYO': '#66cc66',
        'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA': '#ff9966'
        # Add more colors for different departments as needed
    }
    for idx, row in filtered_data.iterrows():
        department_color = color_map.get(row['Department'], 'gray')
        radius = 0.1 if initial else 3  # Smaller radius initially, larger after selections
        folium.CircleMarker(
            location=[row['LATITUD'], row['LONGITUD']],
            radius=radius,  # Adjust the radius
            color=department_color,
            fill=True,
            fill_color=department_color,
            fill_opacity=0.6 if initial else 1,
            opacity=0.3 if initial else 1,  # Make the initial markers less visible
            popup=folium.Popup(f"""
                <b>Municipality:</b> {row['MUNICIPIO']}<br>
                <b>Total Cases:</b> {row['Total']}<br>
                <b>Male Cases:</b> {row['Counts Male']}<br>
                <b>Female Cases:</b> {row['Counts Female']}<br>
                <b>Department:</b> {row['Department']}<br>
                <b>Year:</b> {row['Year']}
            """, max_width=300)
        ).add_to(marker_cluster)
    return folium_map._repr_html_()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=True)
server = app.server

# Layout
app.layout = dbc.Container([
    html.H1('Colombian Suicide Trends Dashboard', style={'textAlign': 'center', 'margin-bottom': '20px'}),
    dbc.Tabs([
        dbc.Tab(label='Interactive Map', tab_id='tab-1'),
        dbc.Tab(label='Trend Plots', tab_id='tab-2'),
    ], id='tabs', active_tab='tab-1'),
    html.Div(id='content', style={'margin-top': '20px'})  # Add margin to create space between tabs and content
])

# Callbacks to switch between tabs
@app.callback(
    Output('content', 'children'),
    [Input('tabs', 'active_tab')]
)
def render_tab_content(active_tab):
    if active_tab == 'tab-1':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label('Select Year'),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': 'Select All', 'value': 'all'}] + [{'label': year, 'value': year} for year in sorted(data['Year'].unique())],
                        value='all',
                        placeholder='Select Year',
                        clearable=True
                    )
                ], width=3, style={'margin-bottom': '20px'}),  # Add margin-bottom to create space between dropdowns
                dbc.Col([
                    html.Label('Select Department'),
                    dcc.Dropdown(
                        id='department-dropdown',
                        options=[{'label': 'Select All', 'value': 'all'}] + [{'label': depto, 'value': depto} for depto in data['Department'].unique()],
                        value='all',
                        placeholder='Select Department',
                        clearable=True
                    )
                ], width=6, style={'margin-bottom': '20px'}),  # Increased width to make the dropdown wider
                dbc.Col([
                    html.Label('Select Municipality'),
                    dcc.Dropdown(
                        id='municipality-dropdown',
                        options=[{'label': 'Select All', 'value': 'all'}],
                        value='all',
                        placeholder='Select Municipality',
                        clearable=True
                    )
                ], width=3, style={'margin-bottom': '20px'}),
            ]),
            dbc.Row([
                dbc.Col(html.Iframe(id='map-plot', style={'height': '70vh', 'width': '100%'}), width=12,
                        style={'padding-left': '0px'}),
            ]),
        ])
    elif active_tab == 'tab-2':
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.ButtonGroup([
                    dbc.Button("Trend in Suicides by Year", id="btn-year", n_clicks=0, style={'width': '70%'}),
                    dbc.Button("Trend in Suicides by Year and Gender", id="btn-year-gender", n_clicks=0, style={'width': '70%'}),
                    dbc.Button("Trend in Suicides by Year and Age Group", id="btn-year-age", n_clicks=0, style={'width': '70%'}),
                ], vertical=True, style={'width': '100%', 'padding-right': '10px'}), width=2),
                dbc.Col(html.Img(id='seaborn-plot', style={'width': '80%', 'height': '70vh'}), width=10),
            ]),
        ])

# Callbacks for plots
@app.callback(
    Output('seaborn-plot', 'src'),
    [Input('btn-year', 'n_clicks'),
     Input('btn-year-gender', 'n_clicks'),
     Input('btn-year-age', 'n_clicks')]
)
def update_seaborn_plot(n_clicks_year, n_clicks_year_gender, n_clicks_year_age):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'btn-year'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    plot_type = 'year'
    if button_id == 'btn-year-gender':
        plot_type = 'year_gender'
    elif button_id == 'btn-year-age':
        plot_type = 'year_age'

    image_base64 = create_seaborn_plot(plot_type, data)
    return f'data:image/png;base64,{image_base64}'

@app.callback(
    Output('municipality-dropdown', 'options'),
    [Input('department-dropdown', 'value')]
)
def set_municipalities_options(selected_department):
    if selected_department == 'all':
        return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in data['MUNICIPIO'].unique()]
    elif selected_department:
        municipalities = data[data['Department'] == selected_department]['MUNICIPIO'].unique()
        return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in municipalities]
    return [{'label': 'Select All', 'value': 'all'}]

@app.callback(
    Output('map-plot', 'srcDoc'),
    [Input('year-dropdown', 'value'),
     Input('department-dropdown', 'value'),
     Input('municipality-dropdown', 'value')]
)
def update_map(selected_year, selected_department, selected_municipality):
    filtered_data = data.copy()

    if selected_year != 'all':
        filtered_data = filtered_data[filtered_data['Year'] == selected_year]
    if selected_department != 'all':
        filtered_data = filtered_data[filtered_data['Department'] == selected_department]
    if selected_municipality != 'all':
        filtered_data = filtered_data[filtered_data['MUNICIPIO'] == selected_municipality]

    # Debugging: print the filtered data to console
    print("Filtered Data:")
    print(filtered_data)

    if not filtered_data.empty:
        counts_by_gender = filtered_data.groupby(['Department', 'MUNICIPIO', 'Sexo de la victima']).size().unstack(fill_value=0).reset_index()
        counts_by_gender.columns.name = None
        counts_by_gender = counts_by_gender.rename(columns={'Hombre': 'Counts Male', 'Mujer': 'Counts Female'})

        counts_by_gender['Counts Male'] = counts_by_gender['Counts Male'].astype(int)
        counts_by_gender['Counts Female'] = counts_by_gender['Counts Female'].astype(int)
        counts_by_gender['Total'] = counts_by_gender[['Counts Male', 'Counts Female']].sum(axis=1)

        # Merge the aggregated counts back into the filtered data
        filtered_data = filtered_data.merge(counts_by_gender, on=['Department', 'MUNICIPIO'], how='left')

        # Debugging: print the final filtered data to console
        print("Final Filtered Data:")
        print(filtered_data)

        folium_map = generate_map(filtered_data, initial=False)
        return folium_map

    return generate_map(data, initial=True)

if __name__ == '__main__':
    app.run_server(debug=False, port=8051)


# #final code with colored markers
# import dash
# from dash import dcc, html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import base64
# import io
# import folium
# import os
#
# # Load data
# file_path = 'https://github.com/rmejia41/open_datasets/raw/main/Suicidios_Colombia_2016_2019_merged.xlsx'
# data = pd.read_excel(file_path)
#
# # Correct mismatched data based on DEPARTAMENTO and MUNICIPIO
# def correct_coordinates(data):
#     corrections = {
#         ('Antioquia', 'Medellin'): (6.25184, -75.56359),
#         # Add more corrections here
#     }
#
#     for (dept, mun), (lat, lon) in corrections.items():
#         data.loc[(data['DEPARTAMENTO'] == dept) & (data['MUNICIPIO'] == mun), ['LATITUD', 'LONGITUD']] = lat, lon
#
#     return data
#
# data = correct_coordinates(data)
#
# # Rename columns for English labels
# data = data.rename(columns={
#     'Año del hecho': 'Year',
#     'DEPARTAMENTO': 'Department'
# })
#
# # Function to create Seaborn line plots
# def create_seaborn_plot(plot_type, data):
#     plt.figure(figsize=(8, 5))  # Adjust the size here
#     if plot_type == 'year':
#         data_grouped = data.groupby('Year').size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', marker='o')
#         plt.xticks(data_grouped['Year'])
#         plt.title('Trend in Suicides by Year', loc='right')
#     elif plot_type == 'year_gender':
#         data_grouped = data.groupby(['Year', 'Sexo de la victima']).size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Sexo de la victima', marker='o', palette='Set1')
#         plt.xticks(data_grouped['Year'].unique())
#         plt.title('Trend in Suicides by Year and Gender', loc='right')
#     elif plot_type == 'year_age':
#         data_grouped = data.groupby(['Year', 'Grupo de edad de la victima']).size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Grupo de edad de la victima', marker='o', palette='tab20')
#         plt.xticks(data_grouped['Year'].unique())
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.title('Trend in Suicides by Year and Age Group', loc='right')
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode('utf-8')
#
# # Function to generate Folium map with dynamic markers
# def generate_map(filtered_data, initial=False):
#     folium_map = folium.Map(location=[4.570868, -74.297333], zoom_start=5)
#     color_map = {
#         'ANTIOQUIA': '#1f77b4',
#         'NORTE DE SANTANDER': '#ff7f0e',
#         'META': '#2ca02c',
#         'CHOCO': '#d62728',
#         'HUILA': '#9467bd',
#         'BOLIVAR': '#8c564b',
#         'CUNDINAMARCA': '#e377c2',
#         'CESAR': '#7f7f7f',
#         'SANTANDER': '#bcbd22',
#         'CALDAS': '#17becf',
#         'CASANARE': '#aec7e8',
#         'LA GUAJIRA': '#ffbb78',
#         'VALLE DEL CAUCA': '#98df8a',
#         'NARINO': '#ff9896',
#         'MAGDALENA': '#c5b0d5',
#         'BOYACA': '#c49c94',
#         'TOLIMA': '#f7b6d2',
#         'RISARALDA': '#c7c7c7',
#         'ARAUCA': '#dbdb8d',
#         'CAUCA': '#9edae5',
#         'QUINDIO': '#ff9da7',
#         'ATLANTICO': '#b6d7a8',
#         'CAQUETA': '#d4a5a5',
#         'BOGOTA D.C.': '#ffcc00',
#         'CORDOBA': '#999999',
#         'SUCRE': '#ff6666',
#         'GUAVIARE': '#cc99ff',
#         'VAUPES': '#99cc00',
#         'VICHADA': '#ffccff',
#         'GUAINIA': '#ff9933',
#         'AMAZONAS': '#66cccc',
#         'PUTUMAYO': '#66cc66',
#         'ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA': '#ff9966'
#         # Add more colors for different departments as needed
#     }
#     for idx, row in filtered_data.iterrows():
#         department_color = color_map.get(row['Department'], 'gray')
#         radius = 0.1 if initial else 3  # Smaller radius initially, larger after selections
#         folium.CircleMarker(
#             location=[row['LATITUD'], row['LONGITUD']],
#             radius=radius,  # Adjust the radius
#             color=department_color,
#             fill=True,
#             fill_color=department_color,
#             fill_opacity=0.8,
#             #opacity=0.5 if initial else 1,  # Make the initial markers less visible
#             popup=folium.Popup(f"""
#                 <b>Municipality:</b> {row['MUNICIPIO']}<br>
#                 <b>Total Cases:</b> {row['Total']}<br>
#                 <b>Male Cases:</b> {row['Counts Male']}<br>
#                 <b>Female Cases:</b> {row['Counts Female']}<br>
#                 <b>Department:</b> {row['Department']}<br>
#                 <b>Year:</b> {row['Year']}
#             """, max_width=250)
#         ).add_to(folium_map)
#     return folium_map._repr_html_()
#
# # Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA], suppress_callback_exceptions=True)
# server = app.server
#
# # Layout
# app.layout = dbc.Container([
#     html.H1('Colombian Suicide Trends Dashboard', style={'textAlign': 'center', 'margin-bottom': '20px'}),
#     dbc.Tabs([
#         dbc.Tab(label='Interactive Map', tab_id='tab-1'),
#         dbc.Tab(label='Trend Plots', tab_id='tab-2'),
#     ], id='tabs', active_tab='tab-1'),
#     html.Div(id='content', style={'margin-top': '20px'})  # Add margin to create space between tabs and content
# ])
#
# # Callbacks to switch between tabs
# @app.callback(
#     Output('content', 'children'),
#     [Input('tabs', 'active_tab')]
# )
# def render_tab_content(active_tab):
#     if active_tab == 'tab-1':
#         return html.Div([
#             dbc.Row([
#                 dbc.Col([
#                     html.Label('Select Year'),
#                     dcc.Dropdown(
#                         id='year-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}] + [{'label': year, 'value': year} for year in sorted(data['Year'].unique())],
#                         value='all',
#                         placeholder='Select Year',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),  # Add margin-bottom to create space between dropdowns
#                 dbc.Col([
#                     html.Label('Select Department'),
#                     dcc.Dropdown(
#                         id='department-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}] + [{'label': depto, 'value': depto} for depto
#                                                                              in data['Department'].unique()],
#                         value='all',
#                         placeholder='Select Department',
#                         clearable=True
#                     )
#                 ], width=6, style={'margin-bottom': '20px'}),  # Increased width to make the dropdown wider
#                 dbc.Col([
#                     html.Label('Select Municipality'),
#                     dcc.Dropdown(
#                         id='municipality-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}],
#                         value='all',
#                         placeholder='Select Municipality',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),
#             ]),
#             dbc.Row([
#                 dbc.Col(html.Iframe(id='map-plot', style={'height': '70vh', 'width': '100%'}), width=12, style={'padding-left': '0px'}),
#             ]),
#         ])
#     elif active_tab == 'tab-2':
#         return html.Div([
#             dbc.Row([
#                 dbc.Col(dbc.ButtonGroup([
#                     dbc.Button("Trend in Suicides by Year", id="btn-year", n_clicks=0, style={'width': '70%'}),
#                     dbc.Button("Trend in Suicides by Year and Gender", id="btn-year-gender", n_clicks=0, style={'width': '70%'}),
#                     dbc.Button("Trend in Suicides by Year and Age Group", id="btn-year-age", n_clicks=0, style={'width': '70%'}),
#                 ], vertical=True, style={'width': '100%', 'padding-right': '10px'}), width=2),
#                 dbc.Col(html.Img(id='seaborn-plot', style={'width': '80%', 'height': '70vh'}), width=10),
#             ]),
#         ])
#
# # Callbacks for plots
# @app.callback(
#     Output('seaborn-plot', 'src'),
#     [Input('btn-year', 'n_clicks'),
#      Input('btn-year-gender', 'n_clicks'),
#      Input('btn-year-age', 'n_clicks')]
# )
# def update_seaborn_plot(n_clicks_year, n_clicks_year_gender, n_clicks_year_age):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         button_id = 'btn-year'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#
#     plot_type = 'year'
#     if button_id == 'btn-year-gender':
#         plot_type = 'year_gender'
#     elif button_id == 'btn-year-age':
#         plot_type = 'year_age'
#
#     image_base64 = create_seaborn_plot(plot_type, data)
#     return f'data:image/png;base64,{image_base64}'
#
# @app.callback(
#     Output('municipality-dropdown', 'options'),
#     [Input('department-dropdown', 'value')]
# )
# def set_municipalities_options(selected_department):
#     if selected_department == 'all':
#         return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in data['MUNICIPIO'].unique()]
#     elif selected_department:
#         municipalities = data[data['Department'] == selected_department]['MUNICIPIO'].unique()
#         return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in municipalities]
#     return [{'label': 'Select All', 'value': 'all'}]
#
# @app.callback(
#     Output('map-plot', 'srcDoc'),
#     [Input('year-dropdown', 'value'),
#      Input('department-dropdown', 'value'),
#      Input('municipality-dropdown', 'value')]
# )
# def update_map(selected_year, selected_department, selected_municipality):
#     filtered_data = data.copy()
#
#     if selected_year != 'all':
#         filtered_data = filtered_data[filtered_data['Year'] == selected_year]
#     if selected_department != 'all':
#         filtered_data = filtered_data[filtered_data['Department'] == selected_department]
#     if selected_municipality != 'all':
#         filtered_data = filtered_data[filtered_data['MUNICIPIO'] == selected_municipality]
#
#     # Debugging: print the filtered data to console
#     print("Filtered Data:")
#     print(filtered_data)
#
#     if not filtered_data.empty:
#         counts_by_gender = filtered_data.groupby(['Department', 'MUNICIPIO', 'Sexo de la victima']).size().unstack(fill_value=0).reset_index()
#         counts_by_gender.columns.name = None
#         counts_by_gender = counts_by_gender.rename(columns={'Hombre': 'Counts Male', 'Mujer': 'Counts Female'})
#
#         counts_by_gender['Counts Male'] = counts_by_gender['Counts Male'].astype(int)
#         counts_by_gender['Counts Female'] = counts_by_gender['Counts Female'].astype(int)
#         counts_by_gender['Total'] = counts_by_gender[['Counts Male', 'Counts Female']].sum(axis=1)
#
#         # Merge the aggregated counts back into the filtered data
#         filtered_data = filtered_data.merge(counts_by_gender, on=['Department', 'MUNICIPIO'], how='left')
#
#         # Debugging: print the final filtered data to console
#         print("Final Filtered Data:")
#         print(filtered_data)
#
#         folium_map = generate_map(filtered_data, initial=False)
#         return folium_map
#
#     return generate_map(data, initial=True)
#
# if __name__ == '__main__':
#     app.run_server(debug=False, port=8051)


##final app with gray markers
# import dash
# from dash import dcc, html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import base64
# import io
# import folium
# from dotenv import load_dotenv
# import os
#
# # Set Matplotlib backend to 'Agg' to avoid GUI-related issues
# plt.switch_backend('Agg')
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Load data
# file_path = '/Suicidios_Colombia_2016_2019_merged.xlsx'
# data = pd.read_excel(file_path)
#
# # Correct mismatched data based on DEPARTAMENTO and MUNICIPIO
# def correct_coordinates(data):
#     corrections = {
#         ('Antioquia', 'Medellin'): (6.25184, -75.56359),
#         # Add more corrections here
#     }
#
#     for (dept, mun), (lat, lon) in corrections.items():
#         data.loc[(data['DEPARTAMENTO'] == dept) & (data['MUNICIPIO'] == mun), ['LATITUD', 'LONGITUD']] = lat, lon
#
#     return data
#
# data = correct_coordinates(data)
#
# # Rename columns for English labels
# data = data.rename(columns={
#     'Año del hecho': 'Year',
#     'DEPARTAMENTO': 'Department'
# })
#
# # Function to create Seaborn line plots
# def create_seaborn_plot(plot_type, data):
#     plt.figure(figsize=(8, 5))  # Adjust the size here
#     if plot_type == 'year':
#         data_grouped = data.groupby('Year').size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', marker='o')
#         plt.xticks(data_grouped['Year'])
#         plt.title('Trend in Suicides by Year', loc='right')
#     elif plot_type == 'year_gender':
#         data_grouped = data.groupby(['Year', 'Sexo de la victima']).size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Sexo de la victima', marker='o', palette='Set1')
#         plt.xticks(data_grouped['Year'].unique())
#         plt.title('Trend in Suicides by Year and Gender', loc='right')
#     elif plot_type == 'year_age':
#         data_grouped = data.groupby(['Year', 'Grupo de edad de la victima']).size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Grupo de edad de la victima', marker='o', palette='tab20')
#         plt.xticks(data_grouped['Year'].unique())
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.title('Trend in Suicides by Year and Age Group', loc='right')
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode('utf-8')
#
# # Function to generate Folium map with dynamic markers
# def generate_map(filtered_data, initial=False):
#     folium_map = folium.Map(location=[4.570868, -74.297333], zoom_start=5)
#     color_map = {
#         'Antioquia': 'blue',
#         'Cundinamarca': 'green',
#         'Bogotá': 'red',
#         'Valle del Cauca': 'purple',
#         'Atlántico': 'orange',
#         'Bolívar': 'darkred',
#         'Boyacá': 'darkblue',
#         'Caldas': 'darkgreen',
#         'Caquetá': 'cadetblue',
#         'Casanare': 'darkpurple',
#         'Cauca': 'lightblue',
#         'Cesar': 'lightgreen',
#         'Chocó': 'lightred',
#         'Córdoba': 'lightgray',
#         'Guainía': 'darkorange',
#         'Guaviare': 'lightpurple',
#         'Huila': 'black',
#         'La Guajira': 'pink',
#         'Magdalena': 'lightbrown',
#         'Meta': 'lightpink',
#         'Nariño': 'beige',
#         'Norte de Santander': 'darkpink',
#         'Putumayo': 'darkbrown',
#         'Quindío': 'khaki',
#         'Risaralda': 'olive',
#         'Santander': 'lightyellow',
#         'Sucre': 'lightcyan',
#         'Tolima': 'lightblue',
#         'Vaupés': 'lightcoral',
#         'Vichada': 'darkgreen'
#         # Add more colors for different departments as needed
#     }
#     for idx, row in filtered_data.iterrows():
#         department_color = color_map.get(row['Department'], 'gray')
#         radius = 0.5 if initial else 2  # Smaller radius initially, larger after selections
#         folium.CircleMarker(
#             location=[row['LATITUD'], row['LONGITUD']],
#             radius=radius,  # Adjust the radius
#             color=department_color,
#             fill=True,
#             fill_color=department_color,
#             fill_opacity=0.6,
#             popup=folium.Popup(f"""
#                 <b>Municipality:</b> {row['MUNICIPIO']}<br>
#                 <b>Total Cases:</b> {row['Total']}<br>
#                 <b>Male Cases:</b> {row['Counts Male']}<br>
#                 <b>Female Cases:</b> {row['Counts Female']}<br>
#                 <b>Department:</b> {row['Department']}<br>
#                 <b>Year:</b> {row['Year']}
#             """, max_width=300)
#         ).add_to(folium_map)
#     return folium_map._repr_html_()
#
# # Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)
#
# # Layout
# app.layout = dbc.Container([
#     html.H1('Colombian Suicide Trends Dashboard', style={'textAlign': 'center', 'margin-bottom': '20px'}),
#     dbc.Tabs([
#         dbc.Tab(label='Interactive Map', tab_id='tab-1'),
#         dbc.Tab(label='Trend Plots', tab_id='tab-2'),
#     ], id='tabs', active_tab='tab-1'),
#     html.Div(id='content', style={'margin-top': '20px'})  # Add margin to create space between tabs and content
# ])
#
# # Callbacks to switch between tabs
# @app.callback(
#     Output('content', 'children'),
#     [Input('tabs', 'active_tab')]
# )
# def render_tab_content(active_tab):
#     if active_tab == 'tab-1':
#         return html.Div([
#             dbc.Row([
#                 dbc.Col([
#                     html.Label('Select Year'),
#                     dcc.Dropdown(
#                         id='year-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}] + [{'label': year, 'value': year} for year in sorted(data['Year'].unique())],
#                         value='all',
#                         placeholder='Select Year',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),  # Add margin-bottom to create space between dropdowns
#                 dbc.Col([
#                     html.Label('Select Department'),
#                     dcc.Dropdown(
#                         id='department-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}] + [{'label': depto, 'value': depto} for depto in data['Department'].unique()],
#                         value='all',
#                         placeholder='Select Department',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),
#                 dbc.Col([
#                     html.Label('Select Municipality'),
#                     dcc.Dropdown(
#                         id='municipality-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}],
#                         value='all',
#                         placeholder='Select Municipality',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),
#             ]),
#             dbc.Row([
#                 dbc.Col(html.Iframe(id='map-plot', style={'height': '70vh', 'width': '100%'}), width=12, style={'padding-left': '0px'}),
#             ]),
#         ])
#     elif active_tab == 'tab-2':
#         return html.Div([
#             dbc.Row([
#                 dbc.Col(dbc.ButtonGroup([
#                     dbc.Button("Trend in Suicides by Year", id="btn-year", n_clicks=0, style={'width': '70%'}),
#                     dbc.Button("Trend in Suicides by Year and Gender", id="btn-year-gender", n_clicks=0, style={'width': '70%'}),
#                     dbc.Button("Trend in Suicides by Year and Age Group", id="btn-year-age", n_clicks=0, style={'width': '70%'}),
#                 ], vertical=True, style={'width': '100%', 'padding-right': '10px'}), width=2),
#                 dbc.Col(html.Img(id='seaborn-plot', style={'width': '80%', 'height': '70vh'}), width=10),
#             ]),
#         ])
#
# # Callbacks for plots
# @app.callback(
#     Output('seaborn-plot', 'src'),
#     [Input('btn-year', 'n_clicks'),
# Input('btn-year-gender', 'n_clicks'),
#      Input('btn-year-age', 'n_clicks')]
# )
# def update_seaborn_plot(n_clicks_year, n_clicks_year_gender, n_clicks_year_age):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         button_id = 'btn-year'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#
#     plot_type = 'year'
#     if button_id == 'btn-year-gender':
#         plot_type = 'year_gender'
#     elif button_id == 'btn-year-age':
#         plot_type = 'year_age'
#
#     image_base64 = create_seaborn_plot(plot_type, data)
#     return f'data:image/png;base64,{image_base64}'
#
# @app.callback(
#     Output('municipality-dropdown', 'options'),
#     [Input('department-dropdown', 'value')]
# )
# def set_municipalities_options(selected_department):
#     if selected_department == 'all':
#         return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in data['MUNICIPIO'].unique()]
#     elif selected_department:
#         municipalities = data[data['Department'] == selected_department]['MUNICIPIO'].unique()
#         return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in municipalities]
#     return [{'label': 'Select All', 'value': 'all'}]
#
# @app.callback(
#     Output('map-plot', 'srcDoc'),
#     [Input('year-dropdown', 'value'),
#      Input('department-dropdown', 'value'),
#      Input('municipality-dropdown', 'value')]
# )
# def update_map(selected_year, selected_department, selected_municipality):
#     filtered_data = data.copy()
#
#     if selected_year != 'all':
#         filtered_data = filtered_data[filtered_data['Year'] == selected_year]
#     if selected_department != 'all':
#         filtered_data = filtered_data[filtered_data['Department'] == selected_department]
#     if selected_municipality != 'all':
#         filtered_data = filtered_data[filtered_data['MUNICIPIO'] == selected_municipality]
#
#     # Debugging: print the filtered data to console
#     print("Filtered Data:")
#     print(filtered_data)
#
#     if not filtered_data.empty:
#         counts_by_gender = filtered_data.groupby(['Department', 'MUNICIPIO', 'Sexo de la victima']).size().unstack(fill_value=0).reset_index()
#         counts_by_gender.columns.name = None
#         counts_by_gender = counts_by_gender.rename(columns={'Hombre': 'Counts Male', 'Mujer': 'Counts Female'})
#
#         counts_by_gender['Counts Male'] = counts_by_gender['Counts Male'].astype(int)
#         counts_by_gender['Counts Female'] = counts_by_gender['Counts Female'].astype(int)
#         counts_by_gender['Total'] = counts_by_gender[['Counts Male', 'Counts Female']].sum(axis=1)
#
#         # Merge the aggregated counts back into the filtered data
#         filtered_data = filtered_data.merge(counts_by_gender, on=['Department', 'MUNICIPIO'], how='left')
#
#         # Debugging: print the final filtered data to console
#         print("Final Filtered Data:")
#         print(filtered_data)
#
#         folium_map = generate_map(filtered_data, initial=False)
#         return folium_map
#
#     return generate_map(data, initial=True)
#
# if __name__ == '__main__':
#     app.run_server(debug=False)




##Original map mapbox
# import matplotlib
# matplotlib.use('Agg')
#
# import dash
# from dash import dcc, html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import base64
# import io
# import plotly.express as px
#
# # Load data
# #file_path = 'https://github.com/rmejia41/open_datasets/raw/main/Suicidios_Colombia_2016_2019_merged.xlsx'
# file_path = 'C:/Users/Biu9/OneDrive - CDC/Python files/Colombia/Data/Suicidios_Colombia_2016_2019_merged.xlsx'
# data = pd.read_excel(file_path)
#
# # Correct mismatched data based on DEPARTAMENTO and MUNICIPIO
# def correct_coordinates(data):
#     corrections = {
#         ('Antioquia', 'Medellin'): (6.25184, -75.56359),
#         # Add more corrections here
#     }
#
#     for (dept, mun), (lat, lon) in corrections.items():
#         data.loc[(data['DEPARTAMENTO'] == dept) & (data['MUNICIPIO'] == mun), ['LATITUD', 'LONGITUD']] = lat, lon
#
#     return data
#
# data = correct_coordinates(data)
#
# # Rename columns for English labels
# data = data.rename(columns={
#     'Año del hecho': 'Year',
#     'DEPARTAMENTO': 'Department'
# })
#
# # Function to create Seaborn line plots
# def create_seaborn_plot(plot_type, data):
#     plt.figure(figsize=(8, 5))  # Adjust the size here
#     if plot_type == 'year':
#         data_grouped = data.groupby('Year').size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', marker='o')
#         plt.xticks(data_grouped['Year'])
#         plt.title('Trend in Suicides by Year', loc='right')
#     elif plot_type == 'year_gender':
#         data_grouped = data.groupby(['Year', 'Sexo de la victima']).size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Sexo de la victima', marker='o', palette='Set1')
#         plt.xticks(data_grouped['Year'].unique())
#         plt.title('Trend in Suicides by Year and Gender', loc='right')
#     elif plot_type == 'year_age':
#         data_grouped = data.groupby(['Year', 'Grupo de edad de la victima']).size().reset_index(name='counts')
#         sns.lineplot(data=data_grouped, x='Year', y='counts', hue='Grupo de edad de la victima', marker='o', palette='tab20')
#         plt.xticks(data_grouped['Year'].unique())
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.title('Trend in Suicides by Year and Age Group', loc='right')
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode('utf-8')
#
# # Initialize the Dash app
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
#
# # Layout
# app.layout = dbc.Container([
#     html.H1('Colombian Suicide Trends Dashboard', style={'textAlign': 'center', 'margin-bottom': '20px'}),
#     dbc.Tabs([
#         dbc.Tab(label='Interactive Map', tab_id='tab-1'),
#         dbc.Tab(label='Trend Plots', tab_id='tab-2'),
#     ], id='tabs', active_tab='tab-1'),
#     html.Div(id='content', style={'margin-top': '20px'})  # Add margin to create space between tabs and content
# ])
#
# # Callbacks to switch between tabs
# @app.callback(
#     Output('content', 'children'),
#     [Input('tabs', 'active_tab')]
# )
# def render_tab_content(active_tab):
#     if active_tab == 'tab-1':
#         return html.Div([
#             dbc.Row([
#                 dbc.Col([
#                     html.Label('Select Year'),
#                     dcc.Dropdown(
#                         id='year-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}] + [{'label': year, 'value': year} for year in sorted(data['Year'].unique())],
#                         value='all',
#                         placeholder='Select Year',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),  # Add margin-bottom to create space between dropdowns
#                 dbc.Col([
#                     html.Label('Select Department'),
#                     dcc.Dropdown(
#                         id='department-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}] + [{'label': depto, 'value': depto} for depto in data['Department'].unique()],
#                         value='all',
#                         placeholder='Select Department',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),
#                 dbc.Col([
#                     html.Label('Select Municipality'),
#                     dcc.Dropdown(
#                         id='municipality-dropdown',
#                         options=[{'label': 'Select All', 'value': 'all'}],
#                         value='all',
#                         placeholder='Select Municipality',
#                         clearable=True
#                     )
#                 ], width=3, style={'margin-bottom': '20px'}),
#             ]),
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id='map-plot', style={'height': '70vh', 'width': '100%'}), width=12, style={'padding-left': '0px'}),
#             ]),
#         ])
#     elif active_tab == 'tab-2':
#         return html.Div([
#             dbc.Row([
#                 dbc.Col(dbc.ButtonGroup([
#                     dbc.Button("Trend in Suicides by Year", id="btn-year", n_clicks=0, style={'width': '70%'}),
#                     dbc.Button("Trend in Suicides by Year and Gender", id="btn-year-gender", n_clicks=0, style={'width': '70%'}),
#                     dbc.Button("Trend in Suicides by Year and Age Group", id="btn-year-age", n_clicks=0, style={'width': '70%'}),
#                 ], vertical=True, style={'width': '100%', 'padding-right': '10px'}), width=2),
#                 dbc.Col(html.Img(id='seaborn-plot', style={'width': '80%', 'height': '70vh'}), width=10),
#             ]),
#         ])
#
# # Callbacks for plots
# @app.callback(
#     Output('seaborn-plot', 'src'),
#     [Input('btn-year', 'n_clicks'),
#      Input('btn-year-gender', 'n_clicks'),
#      Input('btn-year-age', 'n_clicks')]
# )
# def update_seaborn_plot(n_clicks_year, n_clicks_year_gender, n_clicks_year_age):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         button_id = 'btn-year'
#     else:
#         button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#
#     plot_type = 'year'
#     if button_id == 'btn-year-gender':
#         plot_type = 'year_gender'
#     elif button_id == 'btn-year-age':
#         plot_type = 'year_age'
#
#     image_base64 = create_seaborn_plot(plot_type, data)
#     return f'data:image/png;base64,{image_base64}'
#
# @app.callback(
#     Output('municipality-dropdown', 'options'),
#     [Input('department-dropdown', 'value')]
# )
# def set_municipalities_options(selected_department):
#     if selected_department == 'all':
#         return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in data['MUNICIPIO'].unique()]
#     elif selected_department:
#         municipalities = data[data['Department'] == selected_department]['MUNICIPIO'].unique()
#         return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in municipalities]
#     return [{'label': 'Select All', 'value': 'all'}]
#
# @app.callback(
#     Output('map-plot', 'figure'),
#     [Input('year-dropdown', 'value'),
#      Input('department-dropdown', 'value'),
#      Input('municipality-dropdown', 'value')]
# )
# def update_map(selected_year, selected_department, selected_municipality):
#     filtered_data = data.copy()
#
#     if selected_year != 'all':
#         filtered_data = filtered_data[filtered_data['Year'] == selected_year]
#     if selected_department != 'all':
#         filtered_data = filtered_data[filtered_data['Department'] == selected_department]
#     if selected_municipality != 'all':
#         filtered_data = filtered_data[filtered_data['MUNICIPIO'] == selected_municipality]
#
#     # Create aggregated data for counts by gender and total counts
#     if not filtered_data.empty:
#         counts_by_gender = filtered_data.groupby(['Department', 'MUNICIPIO', 'Sexo de la victima']).size().unstack(fill_value=0).reset_index()
#         counts_by_gender.columns.name = None
#         counts_by_gender = counts_by_gender.rename(columns={'Hombre': 'Counts Male', 'Mujer': 'Counts Female'})
#
#         counts_by_gender['Counts Male'] = counts_by_gender['Counts Male'].astype(int)
#         counts_by_gender['Counts Female'] = counts_by_gender['Counts Female'].astype(int)
#         counts_by_gender['Total'] = counts_by_gender[['Counts Male', 'Counts Female']].sum(axis=1)
#
#         # Merge the aggregated counts back into the filtered data
#         filtered_data = filtered_data.merge(counts_by_gender, on=['Department', 'MUNICIPIO'], how='left')
#
#     fig = px.scatter_mapbox(
#      filtered_data,
#      lat='LATITUD',
#      lon='LONGITUD',
#      hover_name='MUNICIPIO',
#      hover_data={
#         'Year': True,
#         'Counts Male': True,
#         'Counts Female': True,
#         'Total': True,
#         'Department': True
#       },
#      color='Department',
#      zoom=5,
#      height=750
#     )
#     fig.update_traces(marker=dict(size=8, symbol='circle', opacity=0.7))
#     fig.update_layout(mapbox_style="carto-positron")
#     return fig
#
# if __name__ == '__main__':
#     app.run_server(debug=False)

