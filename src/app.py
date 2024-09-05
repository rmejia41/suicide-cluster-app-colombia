# Complete map with marker clusters
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
import os

# Set Matplotlib backend to 'Agg' to avoid GUI-related issues
plt.switch_backend('Agg')

# Load data
file_path = 'https://github.com/rmejia41/open_datasets/raw/main/Suicidios_Colombia_2016_2019_merged.xlsx'
data = pd.read_excel(file_path)

# Correct mismatched data based on DEPARTAMENTO and MUNICIPIO
def correct_coordinates(data):
    corrections = {
        ('Antioquia', 'Medellin'): (6.25184, -75.56359),
        # Add more corrections here as needed
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
    plt.figure(figsize=(8, 5))
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
        # Add more colors for different departments as needed
    }
    for idx, row in filtered_data.iterrows():
        department_color = color_map.get(row['Department'], 'gray')
        radius = 0.1 if initial else 3
        folium.CircleMarker(
            location=[row['LATITUD'], row['LONGITUD']],
            radius=radius,
            color=department_color,
            fill=True,
            fill_color=department_color,
            fill_opacity=0.6 if initial else 1,
            opacity=0.3 if initial else 1,
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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE], suppress_callback_exceptions=False)
server = app.server

# Layout
app.layout = dbc.Container([
    html.H1('Colombian Suicide Trends Dashboard', style={'textAlign': 'center', 'margin-bottom': '20px'}),
    dbc.Tabs([
        dbc.Tab(label='Interactive Map', tab_id='tab-1'),
        dbc.Tab(label='Trend Plots', tab_id='tab-2'),
    ], id='tabs', active_tab='tab-1'),
    html.Div(id='content', style={'margin-top': '20px'})
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
                        clearable=True
                    )
                ], width=3),
                dbc.Col([
                    html.Label('Select Department'),
                    dcc.Dropdown(
                        id='department-dropdown',
                        options=[{'label': 'Select All', 'value': 'all'}] + [{'label': dept, 'value': dept} for dept in data['Department'].unique()],
                        value='all',
                        clearable=True
                    )
                ], width=6),
                dbc.Col([
                    html.Label('Select Municipality'),
                    dcc.Dropdown(
                        id='municipality-dropdown',
                        options=[{'label': 'Select All', 'value': 'all'}],
                        value='all',
                        clearable=True
                    )
                ], width=3),
            ]),
            dbc.Row([
                dbc.Col(html.Iframe(id='map-plot', style={'height': '70vh', 'width': '100%'}), width=12),
            ]),
        ])
    elif active_tab == 'tab-2':
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.ButtonGroup([
                    dbc.Button("Trend in Suicides by Year", id="btn-year", n_clicks=0),
                    dbc.Button("Trend in Suicides by Year and Gender", id="btn-year-gender", n_clicks=0),
                    dbc.Button("Trend in Suicides by Year and Age Group", id="btn-year-age", n_clicks=0),
                ], vertical=True), width=2),
                dbc.Col(html.Img(id='seaborn-plot', style={'width': '80%', 'height': '70vh'}), width=10),
            ]),
        ])

# Callbacks for Seaborn plots
@app.callback(
    Output('seaborn-plot', 'src'),
    [Input('btn-year', 'n_clicks'),
     Input('btn-year-gender', 'n_clicks'),
     Input('btn-year-age', 'n_clicks')]
)
def update_seaborn_plot(n_clicks_year, n_clicks_year_gender, n_clicks_year_age):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'btn-year'

    plot_type = 'year'
    if button_id == 'btn-year-gender':
        plot_type = 'year_gender'
    elif button_id == 'btn-year-age':
        plot_type = 'year_age'

    image_base64 = create_seaborn_plot(plot_type, data)
    return f'data:image/png;base64,{image_base64}'

# Callback to populate municipality dropdown based on selected department
@app.callback(
    Output('municipality-dropdown', 'options'),
    [Input('department-dropdown', 'value')]
)
def set_municipalities_options(selected_department):
    if selected_department == 'all':
        return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in data['MUNICIPIO'].unique()]
    else:
        municipalities = data[data['Department'] == selected_department]['MUNICIPIO'].unique()
        return [{'label': 'Select All', 'value': 'all'}] + [{'label': municipality, 'value': municipality} for municipality in municipalities]

# Callback to update the map based on the selected filters
@app.callback(
    Output('map-plot', 'srcDoc'),
    [Input('year-dropdown', 'value'),
     Input('department-dropdown', 'value'),
     Input('municipality-dropdown', 'value')]
)
def update_map(selected_year, selected_department,selected_municipality):
    filtered_data = data.copy()

    if selected_year != 'all':
        filtered_data = filtered_data[filtered_data['Year'] == selected_year]
    if selected_department != 'all':
        filtered_data = filtered_data[filtered_data['Department'] == selected_department]
    if selected_municipality != 'all':
        filtered_data = filtered_data[filtered_data['MUNICIPIO'] == selected_municipality]

    if not filtered_data.empty:
        counts_by_gender = filtered_data.groupby(['Department', 'MUNICIPIO', 'Sexo de la victima']).size().unstack(fill_value=0).reset_index()
        counts_by_gender.columns.name = None
        counts_by_gender = counts_by_gender.rename(columns={'Hombre': 'Counts Male', 'Mujer': 'Counts Female'})

        counts_by_gender['Counts Male'] = counts_by_gender['Counts Male'].astype(int)
        counts_by_gender['Counts Female'] = counts_by_gender['Counts Female'].astype(int)
        counts_by_gender['Total'] = counts_by_gender[['Counts Male', 'Counts Female']].sum(axis=1)

        # Merge the aggregated counts back into the filtered data
        filtered_data = filtered_data.merge(counts_by_gender, on=['Department', 'MUNICIPIO'], how='left')

        # Generate the map with filtered data
        folium_map = generate_map(filtered_data, initial=False)
        return folium_map

    # If no data is selected, generate the initial map
    return generate_map(data, initial=True)

if __name__ == '__main__':
    app.run_server(debug=False)

