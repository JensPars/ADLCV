from glob import glob
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity

d = {}
path2clip = glob("img_embeds/*")
for dir in path2clip:
    paths = glob(f"{dir}/*")
    embeds = []
    for path in paths:
        embeds += [np.load(path)]
    try:
        embeds = np.concatenate(embeds)
        d[dir.split("/")[1]] = cosine_similarity(embeds, embeds).mean() #embeds .min(axis=1)
    except:
        pass

#

sorted_indices = np.argsort(list(d.values()))[::-1]
values_sorted = [list(d.values())[i] for i in sorted_indices]
classes_sorted = [list(d.keys())[i] for i in sorted_indices]
# Create a new figure and set the figure size
fig, ax = plt.subplots(figsize=(40, 20))

# Plot the class frequencies as bars
ax.bar(np.arange(len(classes_sorted)), values_sorted, color='tab:blue', width=0.4)

# Set the tick labels and font size
ax.set_xticks(np.arange(len(classes_sorted))+0.2)
ax.set_xticklabels(classes_sorted, rotation=90, fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Set the labels and title
ax.set_xlabel("", fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Class Frequencies', fontsize=16)
#plt.fig
plt.savefig("barplot.png")



    
#for key, embeds in d.items():
    # Create an AgglomerativeClustering object
    # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, metric="cosine", linkage="single")

    # Fit the model to the embeddings
    # model = model.fit(embeds)

    # Plot the dendrogram
    #Z = sch.linkage(embeds, 'single', metric='cosine')
    #dendrogram(Z, p=)
    # Set the title of the subplot
    #plt.set_title(f"Dendrogram for {key}")
    # Remove the spines of the subplot
    #plt.spines['top'].set_visible(False)
    #plt.spines['right'].set_visible(False)
    #plt.spines['left'].set_position(('outward', 10))
    #plt.spines['bottom'].set_position(('outward', 10))

    # Move the axes to the bottom and left of the plot
    #plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    # Show the plot
    #plt.savefig(f"dendos{key}.png")