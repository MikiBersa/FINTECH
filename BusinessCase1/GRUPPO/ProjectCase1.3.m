close all; clear; clc;

% Step 1: Load the Dataset
data = readtable('./BankClients.xlsx'); % Update with the actual file path

% Assuming you've identified which columns are dummy variables and handled them accordingly
% For demonstration, let's proceed directly to clustering assuming all necessary preprocessing is done

% Prepare the dataset for clustering
% Exclude ID or non-feature columns as appropriate
X = table2array(data(:, 2:end)); % Adjust based on your dataset structure

% Step 2 & 3: Determine the Optimal Number of Clusters using Silhouette Scores
silhouetteScores = [];
kRange = 2:6; % Example range of k values
for k = kRange
    [idx, ~, ~, ~, silh] = kmedoids(X, k, 'Distance', 'euclidean', 'Replicates', 10);
    silhouetteScores(end+1) = mean(silh);
    
    % Plot silhouette for each k
    figure;
    silhouette(X, idx);
    title(sprintf('Silhouette Analysis for k = %d', k));
end

% Determine optimal k based on silhouette scores
[~, optimalKIndex] = max(silhouetteScores);
optimalK = kRange(optimalKIndex);
fprintf('Optimal number of clusters (k) based on silhouette score: %d\n', optimalK);

% Step 4: Perform k-Medoids Clustering with the Optimal k
[optimalIdx, medoidIdx] = kmedoids(X, optimalK, 'Distance', 'euclidean', 'Replicates', 10);


% Step 5: Visualization of Clusters using PCA for dimensionality reduction
[coeff, score] = pca(X);
reducedData = score(:, 1:2); % Keeping the first two principal components

% Plot the clusters
figure;
gscatter(reducedData(:,1), reducedData(:,2), optimalIdx);
title(sprintf('Cluster Visualization with Optimal k = %d', optimalK));
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend('Location', 'best');

