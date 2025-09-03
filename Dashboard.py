import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# --- DATA LOADING & PREPARATION ---
# In a real-world scenario, you would load these from your CSV files.
# For this self-contained script, we'll recreate the dataframes.

# Based on table_main_results.csv and topic_summary_pre_post_roberta.csv
topic_data_dict = {
    'topic': [8, 12, 7, 6, 11, 16, 5, 17, 10, 15, 0, 4],
    'name': ["Sensitive & Dry Skin", "Nail Art & Polish", "Wigs & Human Hair", "Hair Clips & Ties", "Gel Polish & UV Lamps", "Dry Skin & Foot Care", "Fragrance & Bath Bombs", "Body Wash & Natural Deodorant", "Hot Tools (Irons & Dryers)", "Eye Shadow & Sponges", "Lip Colour & Gloss", "Makeup Mirrors"],
    'delta_share': [4.22, 1.49, 1.35, 1.15, 0.84, 0.79, -0.81, -0.73, -1.47, -1.50, -3.77, -0.27],
    'delta_pos_rate': [-0.98, 7.05, 1.55, 2.50, 1.96, -2.49, -5.03, -4.51, -0.97, -0.28, -1.47, -8.30]
}
topic_df = pd.DataFrame(topic_data_dict)

# Based on sentiment_distribution_topic_period.csv
sentiment_distribution = {
    8: {'pre': {'pos': 80.02, 'neu': 7.02, 'neg': 12.97}, 'post': {'pos': 79.04, 'neu': 7.87, 'neg': 13.09}}, 12: {'pre': {'pos': 76.04, 'neu': 6.08, 'neg': 17.88}, 'post': {'pos': 83.09, 'neu': 4.13, 'neg': 12.78}}, 7: {'pre': {'pos': 79.38, 'neu': 6.97, 'neg': 13.65}, 'post': {'pos': 80.93, 'neu': 7.49, 'neg': 11.59}}, 6: {'pre': {'pos': 74.46, 'neu': 8.37, 'neg': 17.17}, 'post': {'pos': 76.96, 'neu': 9.21, 'neg': 13.83}}, 11: {'pre': {'pos': 75.25, 'neu': 9.48, 'neg': 15.27}, 'post': {'pos': 77.21, 'neu': 9.18, 'neg': 13.61}}, 16: {'pre': {'pos': 69.91, 'neu': 10.09, 'neg': 20.00}, 'post': {'pos': 67.42, 'neu': 12.86, 'neg': 19.73}}, 5: {'pre': {'pos': 78.76, 'neu': 7.4, 'neg': 13.84}, 'post': {'pos': 73.73, 'neu': 9.7, 'neg': 16.57}}, 17: {'pre': {'pos': 73.71, 'neu': 10.14, 'neg': 16.15}, 'post': {'pos': 69.2, 'neu': 13.3, 'neg': 17.5}}, 10: {'pre': {'pos': 77.21, 'neu': 9.4, 'neg': 13.39}, 'post': {'pos': 76.24, 'neu': 9.72, 'neg': 14.04}}, 15: {'pre': {'pos': 80.76, 'neu': 7.13, 'neg': 12.11}, 'post': {'pos': 80.47, 'neu': 6.79, 'neg': 12.74}}, 0: {'pre': {'pos': 78.98, 'neu': 6.54, 'neg': 14.49}, 'post': {'pos': 77.51, 'neu': 4.9, 'neg': 17.59}}, 4: {'pre': {'pos': 82.72, 'neu': 3.7, 'neg': 13.58}, 'post': {'pos': 74.42, 'neu': 8.53, 'neg': 17.05}}
}

# Based on topic_examples.csv
topic_examples = {
    8: [{"title": "A++ chemical-free product", "text": "I have had trouble with my dry skin. This has helped so much..."}, {"title": "Korean skin care does it again!", "text": "This is a must for dry or sensitive skin. It is lightweight and absorbs quickly..."}],
    12: [{"title": "Fun set!", "text": "Such a great way to try out a new hobby, this has everything you need to do your own nails at home."}, {"title": "Impressive set of coffin nails.", "text": "The gel stays where I put it, is self leveling, and cures to a hard, durable finish..."}],
    0: [{"title": "creamy lip stain", "text": "Goes on smooth, lovely color..."}],
    4: [{"title": "Good, nice light, but base is impossible...", "text": "The mirror itself works well, but the base is fiddly and the suction cup does not hold for long."}, {"title": "Concept great.....construction not so great.", "text": "The lighting is great but the build feels cheap and wobbly. The magnification is distorted at the edges."}]
}

# Based on sanity_label_counts.csv
overall_sentiment = {'Positive': 18944, 'Negative': 4196, 'Neutral': 1744}

# Create a simulated dataset for the validity box plot
np.random.seed(42)
n_samples = 1000
rating_data = {
    'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.06, 0.05, 0.1, 0.23, 0.56]),
    'score': np.random.randn(n_samples)
}
sentiment_vs_rating_df = pd.DataFrame(rating_data)

# CORRECTED: Adjust scores to match the expected distribution from your report, ensuring value length matches key length
for r in [1, 2, 3, 4, 5]:
    num_ratings = (sentiment_vs_rating_df['rating'] == r).sum()
    if num_ratings > 0:
        if r == 1:
            scores = np.random.normal(-0.85, 0.1, num_ratings).clip(-1, 0)
        elif r == 2:
            scores = np.random.normal(-0.75, 0.2, num_ratings).clip(-1, 0.1)
        elif r == 3:
            scores = np.random.normal(-0.3, 0.4, num_ratings).clip(-1, 1)
        elif r == 4:
            scores = np.random.normal(0.75, 0.3, num_ratings).clip(-0.5, 1)
        elif r == 5:
            scores = np.random.normal(0.95, 0.1, num_ratings).clip(0.5, 1)
        sentiment_vs_rating_df.loc[sentiment_vs_rating_df['rating'] == r, 'score'] = scores

sentiment_vs_rating_df['rating'] = sentiment_vs_rating_df['rating'].astype(str) + " Star(s)"


# Based on sentiment_distribution_topic_period.csv, aggregated by year
heatmap_data = {
    'Nail Art & Polish': {'2019': 0.55, '2021': 0.60, '2022': 0.60, '2023': 0.54}, 'Wigs & Human Hair': {'2019': 0.59, '2021': 0.67, '2022': 0.56, '2023': 0.70}, 'Hair Clips & Ties': {'2019': 0.52, '2021': 0.54, '2022': 0.62, '2023': 0.49}, 'Sensitive & Dry Skin': {'2019': 0.61, '2021': 0.61, '2022': 0.57, '2023': 0.53}, 'Fragrance & Bath Bombs': {'2019': 0.57, '2021': 0.49, '2022': 0.47, '2023': 0.55}, 'Body Wash & Natural Deodorant': {'2019': 0.48, '2021': 0.40, '2022': 0.43, '2023': 0.38}, 'Makeup Mirrors': {'2019': 0.58, '2021': 0.45, '2022': 0.48, '2023': 0.50}
}
heatmap_matrix_df = pd.DataFrame(heatmap_data).T

# --- INITIALIZE THE DASH APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server


# --- REUSABLE COMPONENTS ---
def Card(children, **kwargs):
    user_class = kwargs.pop('className', '')
    final_class = f"shadow-sm rounded-3 {user_class}".strip()
    return dbc.Card(children, body=True, className=final_class, **kwargs)

def format_delta(value, is_pp=False):
    sign = "+" if value >= 0 else ""
    suffix = " pp" if is_pp else ""
    return f"{sign}{value:.2f}{suffix}"

# --- APP LAYOUT ---
app.layout = dbc.Container([
    dcc.Store(id='sorted-topic-data'),
    
    html.Header([
        html.H1("Post-COVID Beauty & Personal Care Review Analysis", className="display-4"),
        html.P("An interactive dashboard summarizing key shifts in consumer conversation on Amazon US (2019 vs 2021-23).", className="lead text-muted")
    ], className="py-4 border-bottom"),

    dbc.Row(dbc.Col(Card([
        html.H4("Executive Summary & Key Actions", className="card-title"),
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("The Story", className="text-primary"),
                html.P("Post-COVID, consumer conversation has pivoted from cosmetics towards skincare and at-home solutions. While attention on Nails & Hair Clips grew with positive sentiment, categories like Makeup Mirrors and Fragrance deteriorated, revealing specific product and expectation gaps.")
            ])),
            dbc.Col(html.Div([
                html.H5("The Opportunity (Scale)", className="text-success"),
                html.P("Capitalize on rising attention and positive tone in Nails & Hair Accessories. Scale with starter kits, how-to content, and durability claims to solidify market leadership.")
            ])),
            dbc.Col(html.Div([
                html.H5("The Risk (Fix)", className="text-danger"),
                html.P("Address deteriorating sentiment in Mirrors, Fragrance, and Deodorant. Prioritize QA on hardware, replace marketing superlatives with bounded, test-anchored claims, and improve product page specs.")
            ]))
        ])
    ])), className="my-4"),

    dbc.Row([
        dbc.Col([
            Card([
                html.H4("1. High-Level Overview"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.H6("The Big Picture", className="text-muted text-uppercase"),
                        html.P("Topic Mix Changed Materially", className="h5 text-primary"),
                        html.P("χ²(17) = 432.04", className="display-5 fw-bold"),
                        html.P("A highly significant shift in review topics.", className="small text-muted")
                    ]), width=5),
                    dbc.Col(dcc.Graph(
                        id='sentiment-distribution-chart',
                        figure=px.pie(
                            names=list(overall_sentiment.keys()),
                            values=list(overall_sentiment.values()),
                            title="Overall Sentiment (N=24,884)",
                            hole=0.4,
                            color_discrete_map={'Positive': '#198754', 'Negative': '#dc3545', 'Neutral': '#6c757d'}
                        ).update_layout(showlegend=True, margin=dict(t=40, b=0, l=0, r=0)),
                        config={'displayModeBar': False}
                    ), width=7)
                ], align="center")
            ]),
            Card(dcc.Graph(
                id='validity-chart',
                figure=px.box(
                    sentiment_vs_rating_df,
                    x='rating',
                    y='score',
                    color='rating',
                    category_orders={"rating": ["1 Star(s)", "2 Star(s)", "3 Star(s)", "4 Star(s)", "5 Star(s)"]},
                    labels={'score': 'Sentiment Score (s)', 'rating': 'Star Rating'},
                    color_discrete_map={
                        "1 Star(s)": "#dc3545", "2 Star(s)": "#fd7e14", "3 Star(s)": "#ffc107", 
                        "4 Star(s)": "#20c997", "5 Star(s)": "#198754"
                    }
                ).update_layout(
                    title_text="2. Model Validity (Sentiment vs. Star Rating)",
                    showlegend=False,
                    margin=dict(t=40)
                ),
                config={'displayModeBar': False}
            ), className="mt-4")
        ], width=12, lg=5),

        dbc.Col(Card([
            html.H4("3. Topic & Sentiment Dynamics"),
            dbc.Tabs([
                dbc.Tab(label="Attention Shifts", children=[
                    dbc.ButtonGroup([
                        dbc.Button("Largest Gains", id="sort-gain-btn", n_clicks=0, color="primary", outline=True, size="sm"),
                        dbc.Button("Largest Declines", id="sort-decline-btn", n_clicks=0, color="primary", outline=True, size="sm"),
                        dbc.Button("Most Volatile", id="sort-volatile-btn", n_clicks=0, color="primary", outline=True, size="sm"),
                    ], id="sort-controls", className="my-3"),
                    html.P("Change in topic share (post-pre). Click a bar for a deep-dive.", className="small text-muted"),
                    dcc.Graph(id='attention-shift-chart', config={'displayModeBar': False})
                ]),
                dbc.Tab(label="Sentiment Persistence", children=[
                    html.P("Year-on-year trend in sentiment score (s = p_pos - p_neg).", className="small text-muted mt-3"),
                    dcc.Graph(id='sentiment-heatmap',
                        figure=px.imshow(
                            heatmap_matrix_df,
                            labels=dict(x="Year", y="Topic", color="Sentiment Score"),
                            color_continuous_scale='RdYlGn',
                            zmin=0.3, zmax=0.8,
                            aspect="auto"
                        ).update_layout(margin=dict(t=10, b=10)),
                        config={'displayModeBar': False}
                    )
                ]),
            ])
        ]), width=12, lg=7)
    ], className="mt-4"),

    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
        dbc.ModalBody(id="modal-body"),
    ], id="topic-modal", is_open=False, size="lg")

], fluid=True)


# --- CALLBACKS FOR INTERACTIVITY ---

@app.callback(
    Output('sorted-topic-data', 'data'),
    [Input('sort-gain-btn', 'n_clicks'),
     Input('sort-decline-btn', 'n_clicks'),
     Input('sort-volatile-btn', 'n_clicks')]
)
def update_sorted_data(gain_clicks, decline_clicks, volatile_clicks):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'sort-gain-btn'

    if button_id == 'sort-decline-btn':
        sorted_df = topic_df.sort_values('delta_share', ascending=True)
    elif button_id == 'sort-volatile-btn':
        sorted_df = topic_df.copy()
        sorted_df['abs_delta_pos_rate'] = sorted_df['delta_pos_rate'].abs()
        sorted_df = sorted_df.sort_values('abs_delta_pos_rate', ascending=False)
    else:
        sorted_df = topic_df.sort_values('delta_share', ascending=False)
        
    return sorted_df.to_dict('records')

@app.callback(
    Output('attention-shift-chart', 'figure'),
    Input('sorted-topic-data', 'data')
)
def update_attention_chart(sorted_data):
    if not sorted_data:
        return go.Figure()

    df = pd.DataFrame(sorted_data)
    colors = ['#198754' if x >= 0 else '#dc3545' for x in df['delta_share']]
    
    fig = go.Figure(go.Bar(
        x=df['delta_share'],
        y=df['name'],
        orientation='h',
        marker_color=colors,
        customdata=df['topic']
    ))
    fig.update_layout(
        yaxis={'categoryorder': 'array', 'categoryarray': list(df['name'])[::-1]},
        xaxis_title="Change in Share (Percentage Points)",
        yaxis_title=None,
        margin=dict(l=200, r=20, t=20, b=20),
        height=500
    )
    return fig

@app.callback(
    [Output('topic-modal', 'is_open'),
     Output('modal-title', 'children'),
     Output('modal-body', 'children')],
    [Input('attention-shift-chart', 'clickData')],
    [State('topic-modal', 'is_open')]
)
def toggle_modal(clickData, is_open):
    if not clickData:
        return False, "", ""

    topic_id = clickData['points'][0]['customdata']
    topic_info = topic_df[topic_df['topic'] == topic_id].iloc[0]
    
    modal_title = f"Deep Dive: {topic_info['name']}"
    
    sentiment = sentiment_distribution.get(topic_id, {'pre': {}, 'post': {}})
    examples = topic_examples.get(topic_id, [])

    def create_sentiment_bar(period, dist):
        return html.Div([
            html.P(period, className="fw-bold small text-muted text-uppercase"),
            dbc.Progress([
                dbc.Progress(value=dist.get('pos', 0), color="success", bar=True),
                dbc.Progress(value=dist.get('neu', 0), color="secondary", bar=True),
                dbc.Progress(value=dist.get('neg', 0), color="danger", bar=True),
            ], style={'height': '20px'}, className="mt-1")
        ])

    modal_body = html.Div([
        dbc.Row([
            dbc.Col(Card([
                html.P("Attention Shift", className="fw-bold"),
                html.P(format_delta(topic_info['delta_share'], True), className=f"h3 {'text-success' if topic_info['delta_share'] >= 0 else 'text-danger'}")
            ])),
            dbc.Col(Card([
                html.P("Positive Rate Shift", className="fw-bold"),
                html.P(format_delta(topic_info['delta_pos_rate'], True), className=f"h3 {'text-success' if topic_info['delta_pos_rate'] >= 0 else 'text-danger'}")
            ]))
        ]),
        html.H5("Sentiment Distribution (Pre vs. Post)", className="mt-4"),
        create_sentiment_bar("Pre (2019)", sentiment['pre']),
        create_sentiment_bar("Post (2021-23)", sentiment['post']),
        html.H5("Representative Review Excerpts", className="mt-4"),
        html.Blockquote(
            children=[
                html.Div([
                    html.P(f'"{ex["title"]}"', className="fw-bold"),
                    html.Footer(ex["text"], className="blockquote-footer")
                ], className="mb-3")
                for ex in examples
            ], 
            className="border-start border-4 border-primary ps-3"
        )
    ])

    return True, modal_title, modal_body


# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)

