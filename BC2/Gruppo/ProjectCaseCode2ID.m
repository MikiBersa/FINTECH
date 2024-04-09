close all;
clc;
clear all;

% Load the datasets from Excel
clients = readtable('Needs.xls', 'Sheet', 'Needs');
products = readtable('Needs.xls', 'Sheet', 'Products');

% Pre-allocate array for storing ideal client IDs for each product
idealClients = zeros(height(products), 1);

% Loop through each product
for i = 1:height(products)
    product = products(i, :);
    
    % Pre-allocate array for distances
    distances = zeros(height(clients), 1);
    
    % Loop through each client to calculate Euclidean distance
    for j = 1:height(clients)
        client = clients(j, :);
        
        % Calculate the squared difference for each feature
        % Adjust 'RiskPropensity', 'Risk', 'Type', 'IncomeInv', and 'AccInv' to your actual column names
        riskDiff = (product.Risk - client.RiskPropensity)^2;
        
        % Assuming 'Type' is a binary feature where '0' matches 'IncomeInv' and '1' matches 'AccInv'
        if product.Type == 0
            typeDiff = (1 - client.IncomeInvestment)^2; % Assuming 'IncomeInv' is binary (1 for match, 0 for no match)
        else
            typeDiff = (1 - client.AccumulationInvestment)^2; % Assuming 'AccInv' is binary (1 for match, 0 for no match)
        end
        
        % Sum squared differences and take square root for Euclidean distance
        distances(j) = sqrt(riskDiff + typeDiff);
    end
    
    % Find the index of the minimum distance
    [~, idx] = min(distances);
    
    % Store the client ID of the ideal client for this product
    idealClients(i) = clients.ID(idx); % Adjust 'ClientID' to your actual client ID column name
end

% Combine product IDs and their ideal client IDs into a table for easy viewing
results = table(products.IDProduct, idealClients, 'VariableNames', {'ProductID', 'IdealClientID'});

% Display the results
disp(results);

% Assuming the previous steps have been completed and 'idealClients' contains the IDs of ideal clients for each product

% Initialize a table to store the ideal client data for each product
idealClientData = table();

% Loop through each product to gather ideal client data
for i = 1:height(products)
    productID = products.IDProduct(i);
    idealClientID = idealClients(i);
    productType = products.Type(i); % Extract the product type for the current product
    
    % Find the row in 'clients' that matches the ideal client ID
    clientData = clients(clients.ID == idealClientID, :);
    
    % Add columns to 'clientData' to indicate the associated product ID and type
    clientData.ProductID = repmat(productID, height(clientData), 1);
    clientData.ProductType = repmat(productType, height(clientData), 1); % Add the product type
    
    % Append this client's data to the 'idealClientData' table
    idealClientData = [idealClientData; clientData];
end

% Display the ideal client data for each product
disp(idealClientData);