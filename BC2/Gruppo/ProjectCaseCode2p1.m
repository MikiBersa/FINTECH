close all;
clc;
clear all;

% Load the first dataset
clients = readtable('Needs.xls', 'Sheet', 'Needs');  % Replace 'Clients' with the actual sheet name

% Load the second dataset
products = readtable('Needs.xls', 'Sheet', 'Products');  % Replace 'Products' with the actual sheet name


% Cartesian Join (Cross Join)
clients.key = ones(height(clients), 1);
products.key = ones(height(products), 1);

% Perform the join
mergedData = outerjoin(clients, products, 'Keys', 'key', 'MergeKeys', true);

% Remove the temporary join key if it exists in mergedData
if ismember('key', mergedData.Properties.VariableNames)
    mergedData.key = [];
end

% Step 2: Remove Duplicate Columns
duplicateColumns = mergedData.Properties.VariableNames(endsWith(mergedData.Properties.VariableNames, '_x') | endsWith(mergedData.Properties.VariableNames, '_y'));
mergedData(:, duplicateColumns) = [];

% Step 3: Preprocess the Data (handle missing values, outliers, etc.)

% Step 4: Add the Flag Variable Based on Given Conditions
% Initialize flag as 0 and find column indices
mergedData.flag = zeros(height(mergedData), 1);
idxIncomeInvestment = find(strcmp('IncomeInvestment', mergedData.Properties.VariableNames));
idxType = find(strcmp('Type', mergedData.Properties.VariableNames));
idxAccumulationInvestment = find(strcmp('AccumulationInvestment', mergedData.Properties.VariableNames));
idxRiskPropensity = find(strcmp('RiskPropensity', mergedData.Properties.VariableNames));
idxRisk = find(strcmp('Risk', mergedData.Properties.VariableNames));

for index = 1:height(mergedData)
    IncomeInvestment = double(mergedData{index, idxIncomeInvestment});
    Type = double(mergedData{index, idxType});
    AccumulationInvestment = double(mergedData{index, idxAccumulationInvestment});
    RiskPropensity = double(mergedData{index, idxRiskPropensity});
    Risk = double(mergedData{index, idxRisk});
    
    if (((IncomeInvestment == 1 & Type == 0) | (AccumulationInvestment == 1 & Type == 1)) & RiskPropensity >= Risk)
        mergedData.flag(index) = 1;
    end
end

% Step 5: Split Data into Training and Validation Sets
cv = cvpartition(size(mergedData,1), 'HoldOut', 0.3);
idxTrain = training(cv);
idxValidation = test(cv);

dataTrain = mergedData(idxTrain,:);
dataValidation = mergedData(idxValidation,:);

% Step 6: Train Logistic Regression Model
predictors = dataTrain{:, [idxIncomeInvestment, idxType, idxAccumulationInvestment, idxRiskPropensity, idxRisk]};
response = dataTrain.flag;
mdl = fitglm(predictors, response, 'Distribution', 'binomial', 'Link', 'logit');

% Step 7: Validate and Evaluate the Model
predictorsValidation = dataValidation{:, [idxIncomeInvestment, idxType, idxAccumulationInvestment, idxRiskPropensity, idxRisk]};
predictions = predict(mdl, predictorsValidation);

% Ensure predictedLabels is of the same type as actualLabels
predictedLabels = double(predictions >= 0.5);

actualLabels = dataValidation.flag;
confusionMat = confusionmat(actualLabels, predictedLabels);
accuracy = sum(diag(confusionMat)) / sum(confusionMat(:));
precision = confusionMat(2,2) / sum(confusionMat(:,2));
recall = confusionMat(2,2) / sum(confusionMat(2,:));
f1Score = 2 * ((precision * recall) / (precision + recall));

% Display the metrics
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1-Score: %.2f\n', f1Score);

coefEstimates = mdl.Coefficients.Estimate;
oddsRatios = exp(coefEstimates);
disp(table(mdl.CoefficientNames.', coefEstimates, oddsRatios, 'VariableNames', {'Predictor', 'Coefficient', 'OddsRatio'}));

[X, Y, T, AUC] = perfcurve(actualLabels, predictions, 1);
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC) ')']);
