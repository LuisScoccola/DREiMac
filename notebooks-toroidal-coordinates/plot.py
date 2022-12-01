import plotly.graph_objects as go
import plotly
import matplotlib as mpl
inline_rc = dict(mpl.rcParams)

import numpy as np


def plot_3d(X, color = None, size = 2, opacity = 1, width = 300, height=200, margins=[0,0,0,0], title="", colorscale="viridis", transparent=False) :
    if colorscale == "viridis":
        cs = plotly.colors.sequential.Viridis
    elif colorscale == "cyclical":
        cs = plotly.colors.cyclical.HSV
    else :
        cs = plotly.colors.diverging.oxy

    if transparent:
        layout = go.Layout( scene=dict( aspectmode='data'), width=width, height=height, margin=go.layout.Margin( l=margins[0], r=margins[1], b=margins[2], t=margins[3]), template="none", title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
    else :
        layout = go.Layout( scene=dict( aspectmode='data'), width=width, height=height, margin=go.layout.Margin( l=margins[0], r=margins[1], b=margins[2], t=margins[3]), template="none", title=title,
        )

    if color is None:
        data_plot = go.Scatter3d(x = X[:,0], y = X[:,1], z = X[:,2], mode='markers', marker=dict(size=size, opacity=opacity))
    else :
        data_plot = go.Scatter3d(x = X[:,0], y = X[:,1], z = X[:,2], mode='markers', marker=dict(size=size,color=color, colorscale=cs, opacity=opacity))

    return go.Figure([data_plot], layout=layout)


# discretize colors for visualizing level sets of a circular coordinate
def disc_col(colors, n_strips = 10, transition = 0.5):
    # transition should be between 0 (very fast) and 1 (slow)
    if transition == 0:
        return [ np.floor(c*n_strips)%2 for c in colors ]
    k = transition
    def sigmoid(x):
        x = (x-0.5) * 2
        s = 1 / (1 + np.exp(-x / k)) 
        return s
    def triangle(y):
        return y if y < 1 else 2 - y
    return [ sigmoid(triangle((c*n_strips)%2)) for c in colors ]
