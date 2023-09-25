##### Experiment 0.0: Memorize mapping from base points to their own idx
> Only a small number of base points(1000) is used. 

This is trivially done by using a single linear layer. loss tends to zero

##### Expperiment 0.1: Memorize the cluster boundaries, given a query and a space, which subspace **this query** should be
> Generate uniform points in the space of base points. Each point is assigned to a cluster (based on distance). Learn the mapping from these points to their clusters (aka nearest neighbor). 
- n_clusters = 1000, n_train = 1997000, n_test = 3000, model = linear 128 x 1000, acc = 0.958
- n_clusters = 100, n_train = 1997000, n_test = 3000, model = linear 128 x 100, acc = 
- n_clusters = 10, n_train = 1997000, n_test = 3000, model = linear 128 x 128, relu, 128 x 10, acc = 0.996
> This is essentially equivalent to doing the distance computation
> Besides learning cluster boundaries(where the query point should be), we need to learn the k-nn distribution(where the neighbors of the points should be). This is not an easy task. 
> The problem of using the previous distance method is that it mistakenly use the distribution of the of query point(based on distance) for its knn distribution

##### Experiment 0.2: Memorize the cluster boundaries, now the clusters are real (instead of each point as a cluster)
> This problem should be essentially the same as Experiment 0.1
> Use 10000 base points, 
- n_base = 10000, n_cluster = 10, n_train = 997000, n_test = 3000, model = linear 128 x 10, acc = 


##### Experiment 1.0: Memorize the cluster boundaries, now predict the cluster of a query's nearest neighbor in base points instead
> Only about 0.6 queries has its cluster equal to the nn cluster (n_clusters=10 (more clusters this number drops), n_base=1000 (more base this number drops(weird, I can see why when base is very small, e.g. 10, this number is high)), n_query=1000(n_query is irrelevant))

It is an easy task to map a query to its closest point (1-linear layer will do)
- mapping to a closest cluster may be harder
- varying it output distribution based on k may be harder
- putting multiple models into one may be harder