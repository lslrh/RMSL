function [nmi,ACC,f,RI,H] = RMSL(X,Z,gt,cls_num,alpha_v, beta, gamma,eta1,eta2, K)
%% Initialize
V = size(X,2); %number of views
N = size(X{1},2); % number of data points
eps = 10e-8;

for v=1:V
    X{v} = X{v}./(repmat(sqrt(sum(X{v}.^2,1)+10e-8),size(X{v},1),1)+eps);
end

for v=1:V
%        Z1{v}=abs(Z{v})+abs(Z{v}');  
%        Z{v} = Z1{v}./(repmat(sqrt(sum(Z1{v}.^2,1)),size(Z1{v},1),1)+eps);
    Z{v} = Z{v}./(repmat(sqrt(sum(Z{v}.^2,1)),size(Z{v},1),1)+eps); 
end
%% initialization
for v=1:V 
    alpha{v} = alpha_v;
    d1{v} = 200;  %dimentionality of the middle layer of BEN
    d2{v} = N;  %dimentionality of the output layer of BEN
end
C = zeros(N,N); 
J = zeros(N,N);
Y = zeros(N,N); H = rand(K,N)/10; %H = H./repmat(sqrt(sum(H.^2,1)),size(H,1),1);
Y2 = zeros(N,N);
%Y3 = zeros(N,N);
Ec = zeros(K,N);
for v=1:V
    d{v}=size(X{v},1);
    W1{v} = rand(d1{v},K)/10; %W1{v} = W1{v}./repmat(sqrt(sum(W1{v}.^2,1)),size(W1{v},1),1)/2;
    W2{v} = zeros(d2{v},d1{v});
    Y1{v}=zeros(N,N);
  % Y4{v}=zeros(N,N);
    Es{v}=rand(d{v},N); 
    R{v} =rand(N,N);
end

%% updating variables...
%gamma =0.1; % regulariation of network
IsConverge = 0;
mu = 1e-5;
pho = 1.5;
max_mu = 1e6;
max_iter = 30;
max_iter_out = 100;
iter = 1;
while (IsConverge == 0&&iter<max_iter_out+1)
    tic
%     eta1 = 0.1; 
%     eta2 = 0.01;
     % Update BEN
     [W1,W2] = ext_updateNetwork(Z,H,W1,W2,alpha,gamma,eta1,max_iter,d1);  

     % Update latent representation H
     H = ext_updateH(Z,H,C,W1,W2,alpha,eta2,max_iter);  

     % Update view-specific subspace representations
     for v = 1:V
        F1{v} = tanh(W1{v}*H);
        Z{v} = inv(X{v}'*X{v}+(mu+alpha{v})*eye(N))*(X{v}'*X{v}+alpha{v}*W2{v}*F1{v}+mu*R{v}-Y1{v});      
        Z1{v}=abs(Z{v})+abs(Z{v}');   
        Z{v} = Z1{v}./(repmat(sqrt(sum(Z1{v}.^2,1)),size(Z1{v},1),1)+eps);  
     end
     
     % Update common subspace representation 
     C = inv(H'*H+mu*eye(N))*(H'*H+mu*J-Y2);
     
     J = softth((C+Y2/mu)+eye(N)*1e-8,beta/mu);
     for v=1:V 
        R{v} = softth((Z{v}+Y1{v}/mu)+eye(N)*1e-8,beta/mu);    
     end 
     
     for v=1:V
        Y1{v} = Y1{v}+ mu.*(Z{v}-R{v});
     end
  
     Y2 = Y2+ mu.*(C-J);
    
     mu = min(pho*mu, max_mu);
    % convergence conditions
    thrsh = 1e-5;
    if(norm(C-J,inf)<thrsh && norm(Z{1}-R{1},inf)<thrsh)
        IsConverge = 1;
    end
%      norm(H-H*C,inf)
    norm(Z{1}-R{1},inf)  
    iter = iter + 1
end
[nmi,ACC,f,RI]=clustering(abs(C)+abs(C'), cls_num, gt);