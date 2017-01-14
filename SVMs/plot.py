from plotly.offline import plot, iplot, init_notebook_mode
from plotly.graph_objs import Scatter
import numpy as np

def scatter(x, y, marker='circle', color='blue', size=5):
    return Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=size,
            symbol=marker,
            color=color,
            opacity=0.8
        )
    )


def line(x, y, color='blue', width=5):
    return Scatter(
        x=x,
        y=y,
        mode='line',
        line=dict(
            width=width,
            color=color,
        )
    )


def plot(plts, title='', x_log=False):
    init_notebook_mode(connected=True)
    fig = dict(
        data=plts,
        layout=dict(
            title=title,
            autosize=True
        )
    )
    if x_log:
        fig['layout']['xaxis'] = dict(type='log')
    iplot(fig)


def plot_classif(X, Y, Ypred, supportvectors=None, title=''):
    true_pos = X[np.where(Y == 1)[0]]
    true_neg = X[np.where(Y == 2)[0]]
    pred_pos = X[np.where(Ypred == 1)[0]]
    pred_neg = X[np.where(Ypred == 2)[0]]
    data = [
        scatter(true_pos[:, 0], true_pos[:, 1]),
        scatter(true_neg[:, 0], true_neg[:, 1], color='red'),
        scatter(pred_pos[:, 0], pred_pos[:, 1], marker='circle-open', size=8),
        scatter(pred_neg[:, 0], pred_neg[:, 1], marker='circle-open', color='red', size=8),
    ]
    if supportvectors is not None:
        data.append(
            scatter(supportvectors[:, 0], supportvectors[:, 1], marker='square-open', color='green', size=10),
        )
    plot(data, title=title)

