import pandas as pd
import numpy as np
from plotly.graph_objs import Bar, Violin

def get_graphs(df):
    '''
    INPUT - (dataframe) data used to generate graphs
    RETURNS - (dict) dictionary to generate plotly JSON graph objects.
    '''

    top10_categories = list(df.iloc[:,3:].sum().sort_values(ascending=False)[:10].index.str.title())
    top10_category_counts = list(df.iloc[:,3:].sum().sort_values(ascending=False)[:10].values)

    graph_one_data = [Bar(
                    x=top10_categories,
                    y=top10_category_counts,
                    marker={'color':'rgb(220,150,170)'}
                )]

    graph_one_layout = {
                'title': 'Top Ten Categories of Messages',
                'yaxis': {'title': "Frequency"},
                'xaxis': {'title': "Category"}
                }

    graph2data = [np.log(len(df['message'][i])) for i in range(df.shape[0])
                    if len(df['message'][i]) > 1]

    graph_two_data = [Violin(x=graph2data,
                        points='outliers',
                        box_visible=True,
                        line_color='black',
                        fillcolor='rgb(150,230,180)')]

    graph_two_layout = {
                'title': 'Length of Messages',
                'yaxis': {'title' : '',
                            'showticklabels' : 'FALSE'},
                'xaxis': {'title': "Number of Characters",
                        'tickmode' : 'array',
                        'tickvals' : [0, 0.6931471806, 1.609437912, 2.302585093,
                        2.995732274, 3.912023005, 4.605170186, 5.298317367,
                        6.214608098, 6.907755279, 7.60090246, 8.517193191,
                        9.210340372, 9.903487553],
                        'ticktext' : ['0', '1', '2', '5', '10', '20', '50',
                        '100', '200', '500', '1000', '2000', '5000', '10000',
                        '20000']}
                        }

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index.str.title())

    graph_three_data = [Bar(
                    y=genre_names,
                    x=genre_counts,
                    orientation='h',
                    marker={'color':'rgb(170,140,230)'}
                )]

    graph_three_layout = {
                'title': 'Message Genres',
                'xaxis': {'title': "Frequency"},
                'yaxis': {'title': "Genre"}
                }

    graphs=[]
    graphs.append(dict(data=graph_one_data, layout=graph_one_layout))
    graphs.append(dict(data=graph_two_data, layout=graph_two_layout))
    graphs.append(dict(data=graph_three_data, layout=graph_three_layout))
    return graphs
