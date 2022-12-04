# ------------------------------------------------------------------------
# InstanceFormer code for visualizing t-sne pplot of the object queries through time.
# ------------------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import plotly.io as pio
import matplotlib.image as mpimg
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

'''
During inference --analysis flag has to be enabled to save the object queries.
'''

pio.renderers.default = 'png'
root = 'root_path'

def plot(fig):
    img_bytes = fig.to_image(format="png")
    fp = io.BytesIO(img_bytes)
    with fp:
        i = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(i, interpolation='nearest')
    plt.show()


def plot_projection(X, comment='', type='tsne'):
    if type=='tsne':
        tsne = TSNE(n_components=2, random_state=0)
        projections = tsne.fit_transform(X.reshape([-1, X.shape[-1]]))
    else:
        umap_2d = UMAP(random_state=0)
        umap_2d.fit(X.reshape([-1,X.shape[-1]]))
        projections = umap_2d.transform(X.reshape([-1,X.shape[-1]]))
    res = []
    for i in list(range(X.shape[0])):
        res.extend([i] * X.shape[1])
    df = pd.DataFrame()
    df["comp-1"] = projections[:, 0]
    df["comp-2"] = projections[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=res,
                    palette=sns.color_palette("hls", X.shape[0]),
                    data=df,legend=False)
    plt.show(dpi=1200)
    # plt.savefig(f'your_root/instanceformer_output/{comment}.png', dpi=1200)

if __name__ == '__main__':
    video = '2a02f752'#'3d04522a'#
    X = np.load(f'{root}/instanceformer_output/analysis/r50_ovis/analysis/{video}_hs_analysis.npy', mmap_mode='r')
    Y = np.load(f'{root}/instanceformer_output/analysis/r50_ovis_tcl/analysis/{video}_hs_analysis.npy', mmap_mode='r')
    X = X[:6,...]
    Y = Y[:6,...]
    plot_projection(X, comment = 'Without Temporal Contrastive Loss', type='tsne')
    plot_projection(Y, comment = 'With Temporal Contrastive Loss', type='tsne')
    # plot_projection(X,'umap')
    # plot_projection(Y,'umap')
