import torch

class TorchKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None, device='cpu'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.device = torch.device(device)
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _init_centroids(self, X):
        n_samples = X.size(0)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        centroid_ids = torch.zeros(self.n_clusters, dtype=torch.long, device=self.device)
        centroid_ids[0] = torch.randint(0, n_samples, (1,), device=self.device)
        centroids = X[centroid_ids[0]].unsqueeze(0)

        for i in range(1, self.n_clusters):
            dists = torch.cdist(X, centroids)
            dists = torch.min(dists, dim=1)[0]

            probs = dists ** 2
            probs = probs / probs.sum()
            centroid_ids[i] = torch.multinomial(probs, 1)
            centroids = torch.cat([centroids, X[centroid_ids[i]].unsqueeze(0)], dim=0)
            
        return centroids
    
    def fit(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)

        centroids = self._init_centroids(X)
        prev_centroids = torch.zeros_like(centroids)

        for iter_num in range(self.max_iter):
            dists = torch.cdist(X, centroids)
            self.labels_ = torch.argmin(dists, dim=1)
            prev_centroids = centroids.clone()
            for k in range(self.n_clusters):
                mask = (self.labels_ == k)
                if mask.sum() > 0:
                    centroids[k] = X[mask].mean(dim=0)

            if torch.norm(centroids - prev_centroids) < self.tol:
                break
                
        self.n_iter_ = iter_num + 1
        self.cluster_centers_ = centroids
        self.inertia_ = torch.sum(torch.min(torch.cdist(X, self.cluster_centers_), dim=1)[0] ** 2).item()
        self.labels_ = self.labels_.cpu().numpy()
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        
        dists = torch.cdist(X, self.cluster_centers_)
        labels = torch.argmin(dists, dim=1)
        return labels.cpu().numpy() 