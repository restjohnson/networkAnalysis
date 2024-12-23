import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx.drawing.nx_pylab import draw_networkx 
import numpy as np
import powerlaw


#dataset
edges = pd.read_csv('emailData.csv', skiprows=1, names=['Source', 'Target', 'Timestamp'])

#directed graph
G = nx.DiGraph()

#to add edges with timestamps as attributes
for _, row in edges.iterrows():
    G.add_edge(row['Source'], row['Target'], timestamp=row['Timestamp'])

#data analysis
in_degree_centrality = nx.in_degree_centrality(G)

out_degree_centrality = nx.out_degree_centrality(G)

degree_centrality = nx.degree_centrality(G)

betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

eigenvector_centrality = nx.eigenvector_centrality(G)

clustering_coefficients = nx.clustering(G)
global_clustering = nx.transitivity(G)

strongly_connected = list(nx.strongly_connected_components(G))
weakly_connected = list(nx.weakly_connected_components(G))

pagerank = nx.pagerank(G, alpha=0.85)

degree_sequence = [degree for _, degree in G.degree()]
degree_count = np.bincount(degree_sequence)
degrees = np.arange(len(degree_count))
degree_sequence = np.array(degree_sequence)
fit = powerlaw.Fit(degree_sequence)
print(f"Power-Law Exponent (alpha): {fit.alpha}")
print(f"Likelihood of Power-Law: {fit.power_law.loglikelihoods}")



"""#emails edges contain timestamps. analysis based for timestamps
# Group edges by timestamp
edges['TimeGroup'] = pd.to_datetime(edges['Timestamp']).dt.date

# Analyze edges for each time group
for time, group in edges.groupby('TimeGroup'):
    subgraph = G.edge_subgraph([(u, v) for u, v in zip(group['Source'], group['Target'])]).copy()"""

metrics = pd.DataFrame({
    'In-Degree Centrality': in_degree_centrality,
    'Out-Degree Centrality': out_degree_centrality,
    'Overall Degreee Centrality': degree_centrality,
    'Betweeness Centrality': betweenness_centrality,
    'Eigenvector Centrality': eigenvector_centrality,
    'Local Clustering Coefficients': clustering_coefficients,
    'Global Clustering Coefficents': global_clustering,
    'Page Rank': pagerank
})
print(metrics)

#Leaders in the field:( Most active senders )
leaders = pd.DataFrame(sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]) #1874
#print(leaders, end="\n")

#Popularity: (Most Emails Received)
popular = pd.DataFrame(sorted(in_degree_centrality.items(), key=lambda x:x[1], reverse=True))[:20] #1874
#print(popular, end="\n")

#Most Active: (Most emails received and sent)
active = pd.DataFrame(sorted(degree_centrality.items(), key=lambda x:x[1], reverse=True))[:20] #1874, #1669, #1159, #453
#print(active, end="\n")

#Information Traansitor
between = pd.DataFrame(sorted(betweenness_centrality.items(), key=lambda x:x[1], reverse=True))[:20] #1669
#print(between, end="\n")

#Individuals with access to the most influential people
influence = pd.DataFrame(sorted(eigenvector_centrality.items(), key=lambda x:x[1], reverse=True))[:20]#1258, #999, #1874
#print(influence, end="\n")

#Key individuals in information dissemination
key_individuals = pd.DataFrame(sorted(pagerank.items(), key=lambda x:x[1], reverse=True))[:20] #1669, #1874, #453
#print(key_individuals, end="\n")

#clustering
clusters = pd.DataFrame(sorted(clustering_coefficients.items(), key=lambda x:x[1], reverse=True))[:20]
#print(clusters)



#print(edges.head())
edges['Timestamp'] = pd.to_datetime(edges['Timestamp'], unit='s')
edges['Date'] = edges['Timestamp'].dt.date


# Group emails by date
email_by_date = edges.groupby('Date').size()
#print(email_by_date)

# Plot activity over time
email_by_date.plot(kind='bar', figsize=(12, 6), title="Email Activity Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Emails")
#plt.show()


node_sizes = [clustering_coefficients[node] * 1000 for node in G.nodes()]
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=False, node_size=node_sizes, node_color='skyblue', edge_color="gray")
plt.title("Nodes Sized by Clustering Coefficients")
#plt.show()


