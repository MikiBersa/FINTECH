close all
clc
clear all

% Load the dataset
data = readtable('Needs.xls');

% Preprocess the data
% Assuming the last column is the target variable and the rest are features
X = data(:, 1:end-1);
Y = data(:, end);

% Convert categorical variables to dummy variables if necessary
% X = dummyvar(X);
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
idx = cv.test;

% Split the data
X_train = X(~idx, :);
Y_train = Y(~idx, :);
disp(X_train);
X_test = X(idx, :);
Y_test = Y(idx, :);

% Train a decision tree model
treeModel = fitctree(X_train, Y_train);

% Predict the responses for the test set
Y_pred = predict(treeModel, X_test);

% Ensure Y_test is an array if it's not already
if istable(Y_test)
    Y_test = table2array(Y_test);
elseif iscategorical(Y_test)
    Y_test = categorical(Y_test);
end

% Do the same for Y_pred if it's not an array
if iscategorical(Y_pred)
    Y_pred = categorical(Y_pred);
end

% Calculate accuracy
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
confMat = confusionmat(Y_test, Y_pred);
disp(confMat);

treeModel = fitctree(X_train, Y_train);
view(treeModel, 'Mode', 'graph');



