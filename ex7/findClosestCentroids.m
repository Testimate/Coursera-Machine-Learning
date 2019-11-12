function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1); 
%%% controids take dim (K*2) that each row is a centroid 

% You need to return the following variables correctly.
m = size(X,1);
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
distance = zeros(K,1);

for iterX = 1:m
    for iterCentroid = 1:K
        XCentroidDiffVec = X(iterX,:) - centroids(iterCentroid,:); 
        distance(iterCentroid) = XCentroidDiffVec * XCentroidDiffVec'; 
    end     
    [~, idx(iterX)] = min(distance);
end





% =============================================================

end

