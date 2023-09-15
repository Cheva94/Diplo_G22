import matplotlib.pyplot as plt
import numpy as np

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


#    fig.savefig(fig_name,dpi=300)
    plt.show()
