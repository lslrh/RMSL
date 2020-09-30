%% RMSL (ICCV-19)
clear;
load('bbcsport_2view.mat');
load('BBC_2view.mat');  % Initialisation of view-specific subspace representations

num_views = size(X,3);
fprintf('Reciprocal multi-layer subspace learning for multi-view clustering\n');
numClust = size(unique(gt),1);
% alpha: view-specific backward encoding networks (BEN)
% beta: regularization on subspace representations
% gamma: regularization on backward encoding networks
% eta1: learning rate for updating networks
% eta2: learning rate for updating latent representation H 
% K: The dimensionality of latent representation 

alpha = 0.4; beta = 0.7; gamma = 0.1; eta1 = 0.01; eta2 = 0.01; K = 200;  % bbcsport
% alpha = 0.6; beta = 0.7; gamma = 0.1; eta1 = 0.1; eta2 = 0.01; K = 200;  % orl

[nmi,ACC,f,RI] = RMSL(X,Z,gt,numClust,alpha,beta,gamma,eta1,eta2, K);
       result=[nmi,ACC,f,RI];
    
