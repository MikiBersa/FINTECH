function D = MixDistance(X,Y)
%
% Mixed categorical/numerical distance 
%
% INPUT:
% X = matrix of features, nObs x (nCategorical + nNumerical)
%        NOTE: categorical features must be
%                               1) grouped together
%                               2) the first block 
% Y = X (for most applications)
%
% OUTPUT:
% D = matrix of distances (nObsCat+nObsNum) x (nObsCat+nObsNum)

%% Find the number of categorical and numerical features
% The idea is that categorical variables are encoded, so they are
% represented by dummy/binary variables,
% and the sum of the possibile values == 1
nFeatures = size(X,2);
nCat = 0;
for i = 1:nFeatures
    % Se questo è vero, significa che tutti i valori nella colonna sono uguali a 1 o 0, indicando che potrebbe essere una variabile categorica binaria.
% Se il controllo sopra è vero, nCat viene incrementato di 1, indicando che è stata trovata un'altra variabile categorica.
    if sum(unique(X(:,i))) == 1
        nCat = nCat + 1;
    end
end
nNum = nFeatures - nCat;

disp(nCat);
disp(nNum);
%% Compute distances, separately
% Calcola le distanze tra le variabili categoriche nelle prime nCat colonne di X e Y utilizzando la distanza di Hamming.
DCat = pdist2(X(:,1:nCat), Y(:,1:nCat), 'hamming');
% Calcola le distanze tra le variabili numeriche nelle colonne da nCat+1 fino alla fine di X e Y utilizzando la distanza di blocco della città (Manhattan).
DNum = pdist2(X(:,nCat+1:end), Y(:,nCat+1:end), 'cityblock');
% Compute relative weight based on the number of categorical variables
wCat = nCat/(nCat + nNum); 
D = wCat*DCat + (1 - wCat)*DNum;

disp(D);
end


