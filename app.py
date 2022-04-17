
import dash
from dash import dcc
from dash import html
import pandas as pd
import base64
import datetime as dt
import plotly.express as px
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go


# ## Import datasets

fpath = 'f1db_csv/'
circuits = pd.read_csv(f'{fpath}circuits.csv', index_col=0, na_values=r'\N')
constructorResults = pd.read_csv(f'{fpath}constructor_results.csv', index_col=0, na_values=r'\N')
constructors = pd.read_csv(f'{fpath}constructors.csv', index_col=0, na_values=r'\N')
constructorStandings = pd.read_csv(f'{fpath}constructor_standings.csv', index_col=0, na_values=r'\N')
drivers = pd.read_csv(f'{fpath}drivers.csv', index_col=0, na_values=r'\N')
driverStandings = pd.read_csv(f'{fpath}driver_standings.csv', index_col=0, na_values=r'\N')
lapTimes = pd.read_csv(f'{fpath}lap_times.csv')
pitStops = pd.read_csv(f'{fpath}pit_stops.csv')
qualifying = pd.read_csv(f'{fpath}qualifying.csv', index_col=0, na_values=r'\N')
races = pd.read_csv(f'{fpath}races.csv', na_values=r'\N')
results = pd.read_csv(f'{fpath}results.csv', index_col=0, na_values=r'\N')
seasons = pd.read_csv(f'{fpath}seasons.csv', index_col=0, na_values=r'\N')
status = pd.read_csv(f'{fpath}status.csv', index_col=0, na_values=r'\N')
iso = pd.read_excel(f'{fpath}ISO3.xlsx', sheet_name='ISO3')
iso.drop(columns ={'NOC'},inplace =True)

circuits = circuits.rename(columns={'name':'circuitName','location':'circuitLocation','country':'circuitCountry','url':'circuitUrl'})
drivers = drivers.rename(columns={'nationality':'driverNationality','url':'driverUrl'})
drivers['driverName'] = drivers['forename']+' '+drivers['surname']
constructors = constructors.rename(columns={'name':'constructorName','nationality':'constructorNationality','url':'constructorUrl'})
races.index = races.index.set_names(['raceId','year','round','circuitId','raceName','date','time','raceUrl','a','b'])
races = races[[]].reset_index()[['raceId','year','round','circuitId','raceName','date','time','raceUrl']]
races.set_index('raceId',inplace=True)
races['date'] = races['date'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))
pitStops = pitStops.rename(columns={'time':'pitTime'})
pitStops['seconds'] = pitStops['milliseconds'].apply(lambda x: x/1000)
results['seconds'] = results['milliseconds'].apply(lambda x: x/1000)


#DADOS
fullnames = drivers['forename'] + str(" ") + drivers['surname']
pilot_names = [dict(label=fullname, value=driver_id) for fullname, driver_id in zip(fullnames, drivers.index)]

newResults = pd.merge(results, races, left_on='raceId', right_index=True, how='left')
newResults = pd.merge(newResults, circuits, left_on='circuitId', right_index=True, how='left')
newResults = pd.merge(newResults, constructors, left_on='constructorId', right_index=True, how='left')
newResults = pd.merge(newResults, drivers, left_on='driverId', right_index=True, how='left')

newPitStops = pd.merge(pitStops, races, left_on='raceId', right_index=True, how='left')
newPitStops = pd.merge(newPitStops, circuits, left_on='circuitId', right_index=True, how='left')
newPitStops = pd.merge(newPitStops,
                       newResults[['raceId', 'driverId', 'driverName', 'constructorId', 'constructorName','points']],
                       left_on=['raceId', 'driverId'], right_on=['raceId', 'driverId'])
constructor_names = [dict(label=constructorName, value=constructorId) for constructorName, constructorId in zip(newPitStops['constructorName'].unique(), newPitStops['constructorId'].unique())]

team_options = [{'label': 'Average Pit Stop Time', 'value': 'PitStop'},
                {'label': 'Points per season', 'value': 'Points'},
                ]

newLapTimes = pd.merge(lapTimes,races,left_on='raceId',right_index=True,how='left')
newLapTimes = pd.merge(newLapTimes,circuits,left_on='circuitId',right_index=True,how='left')
newLapTimes = pd.merge(newLapTimes,newResults[['raceId','driverId','driverName','constructorId','constructorName','points']],left_on=['raceId','driverId'],right_on=['raceId','driverId'])

results_status = pd.merge(results.loc[(results['statusId'] ==  3) | (results['statusId'] ==  4)],races,left_on='raceId',right_index=True,how='left')
results_status = pd.merge(results_status,circuits,left_on='circuitId',right_index=True,how='left')
results_status.rename(columns={'circuitCountry':'Country'},inplace = True)

'''results_status['Country'].loc[results_status['Country'] == 'USA'] = 'United States'
results_status['Country'].loc[results_status['Country'] == 'UAE'] = 'United Arab Emirates'
results_status['Country'].loc[results_status['Country'] == 'UK'] = 'United Kingdom'
results_status['Country'].loc[results_status['Country'] == 'Russia'] = 'Russian Federation'
results_status['Country'].loc[results_status['Country'] == 'China'] = 'China, People\'s Republic of'
results_status['Country'].loc[results_status['Country'] == 'Korea'] = 'Korea, Republic of (South)'''

results_status_GP = results_status.groupby(['circuitLocation','Country']).count()
results_status = results_status.groupby('Country').count()

results_status = pd.merge(results_status,iso,left_on='Country',right_on='Country',how='left')

results_status = results_status.drop_duplicates().reset_index()
results_status_GP = results_status_GP.drop_duplicates().reset_index()

top_5_accidents = results_status_GP.loc[results_status_GP.index.isin(results_status_GP['raceId'].nlargest(5).index)].sort_values(by='raceId', ascending=False)


#TOP 5 CIRCUITS BY ACCIDENTS

table_accidents = go.Table(
        header=dict(
            values=["Circuit ","Country","Total Number of accidents"],
            line_color=['black'],
            fill_color=['rgb(128,128,128)', 'rgb(128,128,128)', 'rgba(138, 8, 8, 0.8)'],
            align='center',
            font=dict(color=['black'], size=15),
        ),
        cells=dict(
            values=[top_5_accidents.circuitLocation,top_5_accidents.Country , top_5_accidents.raceId,],
            line_color='black', fill_color='rgb(43, 43, 43)',
            align=['center','center'], font=dict(color='white', size=14),
            height= 25
        )
    )
layout_table = dict(
        showlegend=False,
        plot_bgcolor='black',
        paper_bgcolor='black',
        autosize=False,
        margin=dict(autoexpand=False, l=10, r=10, t=10, b=5),
        width = 600
    )

table_top5_accidents= go.Figure(data=table_accidents, layout=layout_table)

table_top5_accidents.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })


#CHOROPLETH ACCIDENTS
fig_accidents = px.choropleth(results_status, locations='ISO', color='statusId',
                           color_continuous_scale = 'reds',
                           range_color=(0, 160),
                           labels={'statusId':'Number of Accidents'},
                           hover_name= 'Country'
                              )
fig_accidents.layout.plot_bgcolor = 'rgb(18,19,20)'
fig_accidents.layout.paper_bgcolor = 'rgb(18,19,20)'

fig_accidents.update_geos(
    showcoastlines=True, coastlinecolor='rgb(43, 43, 43)',
    showland=True, landcolor='rgb(43, 43, 43)',
    showocean=True, oceancolor='rgb(30, 30, 30)',
    showlakes=True, lakecolor='rgb(30, 30, 30)',
    showframe=False,
)


fig_accidents.update_layout(margin={"r":10,"t":0,"l":10,"b":0},
                            coloraxis_colorbar=dict(
                                title=dict(text ="Number of Accidents",font = dict(color = 'white')),
                                yanchor="top", y=1,
                                ticks="inside",
                                tickfont = dict(color = 'white')
                            ),
                            dragmode = False,
                            coloraxis_colorbar_x=-0.12
                            )


seasons = [dict(label=year, value=year1) for year, year1 in zip(newLapTimes['year'].sort_values().unique(), newLapTimes['year'].sort_values().unique())]
circuits = [dict(label=name, value=id) for name, id in zip(newLapTimes['circuitLocation'].unique(), newLapTimes['circuitId'].unique())]

encoded_image_avatar = base64.b64encode(open('images/avatar.png', 'rb').read())
encoded_image_logo = base64.b64encode(open('images/f1_logo.png', 'rb').read())


#INTERACTIVE PARTS
    
#LAYOUT   
app = dash.Dash(__name__, assets_folder='style')

server = app.server

app.layout = html.Div([

       html.Div([

           html.Img(src='data:image/png;base64,{}'.format(encoded_image_logo.decode()), id='logo'),


        ], id = 'div_title'),


        html.Div([

        html.H5('Choose your driver', id = 'drivers_title'),

       html.Br(),

       dcc.Dropdown(
        id='names_drop',
        options= pilot_names,
        value=1,
        multi=False
    ),

       html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image_avatar.decode()), id='avatar'),
       ], id = "img_div"),

       html.Div([

       html.P('Nationality: ', id='nationality'),
       html.P('Birthday: ', id='birthday'),
       html.P('Championships: ', id='championships'),
       html.P('Teams: ', id='teams')

       ], id = "info_driver"),

        ], id = 'left_column_drivers',
            style={
                'border': '2px solid black',
                'borderRadius': '15px',
                'overflow': 'hidden'
            }
        ),
    
    html.Div([

    dcc.Graph(
        id='driver_stats_graph',

    )

    ], id = 'right_column_drivers',
        style={
            'border': '2px solid black',
            'borderRadius': '15px',
            'overflow': 'hidden'
        }
    ),

    html.Div([

    html.Hr(style = {'width':'80%',
                     'margin-bottom':'50px'}),

        html.H1('Top 5 circuits with most accidents', style={'margin-left': '60%'}),

        html.H1('Select your teams', id = 'title_teams'),
        html.Div([
            dcc.Graph(figure=table_top5_accidents, style = {'margin-left':'50px','float':'left'}),

        ], style={'width': '50%',
                  'float': 'right',
                  'height':'200px',

                  },

        ),

        html.Div([
    dcc.Dropdown(
        id='team_a_drop',
        options= constructor_names,
        value=1,
        multi=False
    ),

    dcc.Dropdown(
        id='team_b_drop',
        options= constructor_names,
        value=6,
        multi=False
    ),

    dcc.Dropdown(
        id='team_option_drop',
        options= team_options,
        value= 'PitStop',
        multi=False
    ),
    html.Br(),
    html.Br(),
    html.H4("Note: Average pit stop time is only available from 2011 onwards"),



    ], id = "div_dd_teams",
    style = {'width':'50%',
             'height': '100px',
             "margin-top": '4%',
             "margin-left": '2%'
             }),

        html.Div([
    dcc.RangeSlider(
        id='year_slider',
        min=1950,
        max=2022,
        value=[1950, 2022],
        marks={'1950': '1950',
               '1960': '1960',
               '1970': '1970',
               '1980': '1980',
               '1990': '1990',
               '2000': '2000',
               '2010': '2010',
               '2022': '2022'},
        persistence_type='session',
        step=1
         )
    ], id = 'slider_div',
        style ={'width':'40%',
             'margin-left':'80px',
             'border': '2px solid black',
             'borderRadius': '15px',
             'overflow': 'hidden',
             'background-color': 'rgb(18,19,20)',
             'padding-top': '20px',
             'margin-bottom': '20px'
                }),

        html.Div([
    html.H2('Number of accidents and collisions in F1 Races per country (1950-2022)', id='title_choropleth',
                    style= {'margin-left': '52%'})
        ],style= {"height" :'50px','margin-top':' 60px'}),


    html.Div([

        html.Div([
            dcc.Graph(figure=fig_accidents)
        ], style={'width': '48%',
                  'float':'right'}),

    dcc.Graph(
        id='team_stats_graph',
        style = { 'width': '48%',
                'float':'left'}
    )

    ], id = 'teams_graph_div',
        style={
            'borderRadius': '15px',
            'overflow': 'hidden',
            'width': '95%',
            'display': 'block',
            'margin-right': 'auto',
            'margin-left': 'auto'
        }
    )

    ], id = 'teams_div'),

    html.Hr(style={'width': '80%',
                   'margin-bottom': '50px',
                   'margin-top': '50px'}),

html.Div([

    html.H1('Fastest Laps',style = {'display': 'block',
           'margin-right': 'auto',
           'margin-left': '38%'}),
    html.H2('Choose a season and a circuit:',id ='title_season_circuit' ,style = {'display': 'block',
           'margin-right': 'auto',
           'margin-left': '30%'}),
    dcc.Dropdown(
        id='season_drop',
        options= seasons,
        value=2011,
        multi=False
    ),

    dcc.Dropdown(
        id='circuit_drop',
        options= circuits,
        value=11,
        multi=False
    ),

    ], id = "div_dd_circuits",
    style={'display': 'block',
           'margin-right': 'auto',
           'margin-left': 'auto',
           'margin-top':'50px',
           'margin-bottom':'100px',
           'width': '50%'}
    ),

    html.Div([
    dcc.Graph(
        id='circuit_season_graph',
        style={
            'border': '2px solid black',
            'borderRadius': '15px',
            'overflow': 'hidden',
            'width': '95%',
            'display': 'block',
            'margin-right': 'auto',
            'margin-left': 'auto'
        }
    )
    ]),



])

@app.callback(
    Output(component_id='driver_stats_graph', component_property='figure'),
    [Input(component_id='names_drop', component_property='value')]
)
def update_driver_stats(driver):

    season_year_most_wins= "None"

    wins_raceId = results['raceId'].loc[(results['position'] == 1) & (results['driverId'] == driver)].tolist()
    if wins_raceId:
        most_wins_season = races.loc[races.index.isin(wins_raceId)].groupby('year').count().max()['round']
    else:
        most_wins_season = 0

    if most_wins_season > 0:
        season_year_most_wins = races.loc[races.index.isin(wins_raceId)].groupby('year').count()['round'].idxmax()
    x = [len(qualifying.loc[(qualifying['position'] == 1) & (qualifying['driverId'] == driver)]),
         most_wins_season,
         len(results.loc[(results['position'] == 1) & (results['driverId'] == driver)]),
         len(results.loc[results['driverId'] == driver])]

    fig = go.Figure(go.Bar(
        x=x,
        y=[ 'Total Pole Postions','Most wins in a Season (' + str(season_year_most_wins) +")",
            'Total Wins', 'Total Number of Races'],
        marker=dict(
        color='rgba(222, 83, 83, 0.8)',
        line=dict(color='rgba(138, 8, 8, 0.8)', width=3)),
        text= x,
        textposition='outside',
        textfont_color="white",
        orientation='h'))

    fig.layout.plot_bgcolor = 'rgb(18,19,20)'
    fig.layout.paper_bgcolor = 'rgb(18,19,20)'

    driver_name = drivers['driverName'].loc[drivers.index == driver]
    #fig.layout.title= driver_name.iloc[0]
    fig.update_layout(title_text=driver_name.iloc[0], title_x=0.5)




    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            range=[0, 360],


        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
            tickfont=dict(
                family='Trebuchet MS',
                size=18,
                color='white',
            ),
        ),
        margin=dict(
            pad=20
        ),
        showlegend=False,
        title = dict( font = dict(color = 'white'))
    )

    return go.Figure(data=fig)

@app.callback(
     Output(component_id='teams', component_property='children'),
    [Input(component_id='names_drop', component_property='value')]
    )
def update_driver_teams(driver):
    z = results['constructorId'].loc[results['driverId'] == driver].unique()
    u = constructors['constructorName'].loc[constructors.index.isin(z)]
    teams = u.iloc[:].to_list()
    return 'Teams: {}'.format(teams)

@app.callback(
     Output(component_id='championships', component_property='children'),
    [Input(component_id='names_drop', component_property='value')]
    )
def update_driver_championship(driver):
    merged = pd.merge(races, results, left_on='raceId', right_on='raceId', how='right')
    sum_points = merged.groupby(['year', 'driverId']).sum()
    sum_points['YearDriver'] = sum_points.index
    champions = sum_points.groupby('year')['points'].transform(max) == sum_points['points']
    champions_df = pd.DataFrame(data=champions)
    final = np.array(champions_df.index[champions_df['points'] == True])
    final = np.delete(final, -1)

    def column(matrix, i):
        return np.array([row[i] for row in matrix])

    champions_driverId = column(final,1)

    return 'World Championships: {}'.format(np.count_nonzero(champions_driverId == driver))

@app.callback(
     Output(component_id='birthday', component_property='children'),
    [Input(component_id='names_drop', component_property='value')]
    )
def update_driver_birthday(driver):
    birthday = drivers['dob'].loc[drivers.index == driver].iloc[0]
    return 'Birthday: {}'.format(birthday)

@app.callback(
     Output(component_id='nationality', component_property='children'),
    [Input(component_id='names_drop', component_property='value')]
    )
def update_driver_nationality(driver):
    nationality = drivers['driverNationality'].loc[drivers.index == driver].iloc[0]
    return 'Nationality: {}'.format(nationality)


@app.callback(
    Output(component_id='team_stats_graph', component_property='figure'),
    [Input(component_id='team_a_drop', component_property='value')],
    [Input(component_id='team_b_drop', component_property='value')],
    [Input(component_id='team_option_drop', component_property='value')],
    [Input('year_slider', 'value')]
)
def update_team_stats_graph(team_a,team_b,option,year):

    newResults_filtered = newResults[(newResults['year'] >= year[0]) & (newResults['year'] <= year[1])]
    newPitStops_filtered = newPitStops[(newResults['year'] >= year[0]) & (newPitStops['year'] <= year[1])]

    #team points
    if option == 'Points':
        team_stats = newResults_filtered.loc[(newResults['constructorId'] == team_a) | (newResults_filtered['constructorId'] == team_b)].groupby(by=['year', 'constructorName']).sum().reset_index()
        y = 'points'
        title = "Points per Season"
    if option == 'PitStop':
        team_stats = newPitStops_filtered.loc[(newPitStops['seconds'] < 60) & ((newPitStops_filtered['constructorId'] == team_a) | (newPitStops_filtered['constructorId'] == team_b) )].groupby(by=['year', 'constructorName']).mean().reset_index()
        y = 'seconds'
        title = "Average Pit Stop time"


    fig = px.line(team_stats,
                  x='year',
                  y= y,
                  color='constructorName',
                  markers= True,
                  title = title
                  )

    fig.layout.plot_bgcolor = 'rgb(18,19,20)'
    fig.layout.paper_bgcolor = 'rgb(18,19,20)'

    fig.update_layout(hovermode="x unified")
    fig.update_traces(mode="markers+lines", hovertemplate=None)

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=True,
            showticklabels=True,
            zeroline=False,
            tickfont=dict(
                size=13,
                color='white',
            ),
            color= 'white'

        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            showticklabels=True,
            zeroline=False,
            tickfont=dict(
                size=13,
                color='white',
            ),
            color='white'

        ),
        margin=dict(
            pad=20
        ),
        legend=dict(
        title = 'Teams',
        font = dict(
         color = 'white'
        )),
        title=dict(
            font=dict(
                color='white'
            )),
        showlegend=True,
    )

    return go.Figure(data=fig)

@app.callback(
    Output(component_id='circuit_season_graph', component_property='figure'),
    [Input(component_id='season_drop', component_property='value')],
    [Input(component_id='circuit_drop', component_property='value')],
)
def update_season_circuit(season,circuit):
    season_circuit = newLapTimes.loc[(newLapTimes['year'] == season) & (newLapTimes['circuitId'] == circuit)].groupby(by=['driverName']).min().reset_index().sort_values('time_x',ascending=True)
    season_circuit.rename(columns={'time_x': 'Fastest Lap', 'driverName': 'Driver'},inplace=True)

    fig = px.bar(season_circuit, x='Driver', y='Fastest Lap',
                 text='Fastest Lap',
                 )
    fig.update_traces( textposition='outside', textfont_color="white")


    fig.layout.plot_bgcolor = 'rgb(18,19,20)'
    fig.layout.paper_bgcolor = 'rgb(18,19,20)'

    fig.update_traces(marker_color='rgba(222, 83, 83, 0.8)', marker_line_color='rgba(138, 8, 8, 0.8)',
                      marker_line_width=1.5)

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
            tickfont=dict(
                size=11,
                color='white',
            ),
            color='white'

        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            tickfont=dict(
                size=13,
                color='white',
            ),
            color='white'

        ),
        margin=dict(
            pad=20
        ),

        title=dict(
            font=dict(
                color='white'
            )),

    )

    return go.Figure(data=fig)


#
if __name__ == '__main__':
    app.run_server(debug=True)





