import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestCentroid
from random import sample

#-------------------------------------------------------------------------------

def plots(x,y,x_label,y_label,cluster_labels,n_clusters,sample_silhouette_values,silhouette_avg,fig_name):

    fig, [ax1, ax2] = plt.subplots(1,2)
    fig.set_size_inches(18, 7)

    y_lower = 10

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

#        color = cm.nipy_spectral(float(i) / n_clusters)
        color = colors[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
#            facecolor=color,
#            edgecolor=color,
            color = color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xlim(-0.1, 1)
    ax1.set_ylim(0, len(x) + (n_clusters + 1) * 10)

    colors_points = []
    for i in range(len(cluster_labels)):
        for j in range(n_clusters):
            if cluster_labels[i] == j:
                colors_points.append(colors[j])

    #ploteo los puntos
    ax2.scatter(x, y, c=colors_points)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)


    fig.savefig(fig_name,dpi=300)

#-------------------------------------------------------------------------------


path = 'data/fifa2023.csv'
fifa23 = pd.read_csv(path)

fifa23 = fifa23[fifa23['gk']<50]


vars_mod = ['crossing', 'finishing', 'heading', 'short_passing', 'volleys',
            'marking', 'standing_tackle', 'sliding_tackle', 'acceleration',
            'sprint', 'agility', 'balance', 'shot_power', 'stamina',
            'long_shots', 'dribbling', 'curve', 'fk_acc', 'long_passing',
            'ball_control', 'aggression', 'interceptions', 'positioning',
            'vision', 'penalties', 'composure', 'ls', 'st', 'rs', 'lw', 'lf',
            'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm',
            'rm', 'ldm', 'cdm', 'rdm', 'lwb', 'rwb', 'lb', 'lcb', 'cb', 'rcb',
            'rb']

# No resetear índices, para que después sea fácil buscar nombre y demás
fifa_mod = fifa23[vars_mod]/100.    #normalizo por 100 nomas

#-------------------------------------------------------------------------------
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(fifa_mod)
print("Componentes principales:")
#print(pca.components_)

# Llevamos la matriz al espacio de componentes principales
fifa_mod_pca = pca.transform(fifa_mod)

print("Dimensión de la matriz transformada:")
print(fifa_mod_pca.shape)


positions = ['ls', 'st', 'rs', 'lw', 'lf',
'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm',
'rm', 'ldm', 'cdm', 'rdm', 'lwb', 'rwb', 'lb', 'lcb', 'cb', 'rcb',
'rb']

npicks = 3

positions0 = sample(positions,npicks)
nvar = len(positions0)

#ploteo el score de silueta para algunas variables para tener una idea de la clusterización
for nclus in range(2,8):

    for i in range(npicks):
        for j in range(i):

            x = fifa23[positions0[i]]
            y = fifa23[positions0[j]]

            hierarchical_cluster = AgglomerativeClustering(n_clusters=nclus, metric='euclidean', linkage='ward')
            labels = hierarchical_cluster.fit_predict(fifa_mod)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(np.array([x,y]).T, labels)
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(np.array([x,y]).T, labels)

            x_label = positions0[i]
            y_label = positions0[j]
            fig_name = 'tmp/results__nclus'+str(nclus)+'_comun_pos_'+positions0[i]+'_'+positions0[j]

            plots(x,y,x_label,y_label,labels,nclus,sample_silhouette_values,silhouette_avg,fig_name)

#---------------------------------------------------------------------------------------------

#Ahora calculo el score de silueta promedio para cada variable en función del número de clústers:

optimal_n_clus = []
for i in range(len(positions)):

    silhouette_avg_lst = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('avg score of '+positions[i])
    ax.set_ylabel('silouette score')
    ax.set_xlabel('number ofclusters')

    x = fifa23[positions[i]]
    for nclus in range(2,8):
        silhouette_avg=0
        for j in range(len(positions)):
            if i!=j:

                y = fifa23[positions[j]]

                hierarchical_cluster = AgglomerativeClustering(n_clusters=nclus, metric='euclidean', linkage='ward')
                labels = hierarchical_cluster.fit_predict(fifa_mod)

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg += silhouette_score(np.array([x,y]).T, labels)
        silhouette_avg_lst.append(silhouette_avg)
    silhouette_avg_lst = np.array(silhouette_avg_lst)/(len(positions)-1)
    optimal_n_clus.append(np.argmax(silhouette_avg_lst))

    ax.plot(np.linspace(2,7,6,dtype='int'), silhouette_avg_lst, linestyle = '--',linewidth=.7, marker='.',markersize=8)
    fig.savefig('tmp/avg_sil_score_'+str(positions[i]))

print('nro optimo de clusters:',int(optimal_n_clus)+2)
