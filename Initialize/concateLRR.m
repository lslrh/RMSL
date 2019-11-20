function [nmi,ACC,AR,f,p,r,RI,Z] = concateLRR(X,gt,cls_num,lambda1,lambda2,k)
%% Initialize P,X,Z,E_h,E_r,J,F and Y_1,Y_2,Y_3,Y_4
% V1 = size(X,2);  %nunber of views
    V = size(X,2);
    N = size(X{1},2); % number of data points

for i=1:V
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+eps);
end

for i=1:V
%   for i=2:2
    D = size(X,1); % dimension of each view
end
SD = 0;

% M = zeros(N, N);
% for i=1:V
%   M = M+X{i};
% end
M = X{k};

  K=size(M,1);
  X = M;

E = zeros(K,N);   %K是指定的
Z = zeros(N,N); J = zeros(N,N);
Y1 = zeros(K,N);Y2 = zeros(N,N);

IsConverge = 0;    
mu = 1e-5;
%lambda = 100;
pho = 1.1;
max_mu = 1e10;
max_iter =200;
iter = 1;
epsilong = 1e-9;

%% updating variables...
while (IsConverge == 0&&iter<=max_iter)

    % update J

    J = softth((Z+Y2/mu)+eye(N)*1e-8,lambda1/mu);    %lambda1没有定义
 
    % update Z
    Z = inv(X'*X+eye(N))*((J+X'*X-X'*E)+(X'*Y1-Y2)/mu);
    
    % update E
    
    G = [X-X*Z+Y1/mu];
    [E] = solve_l1l2(G,lambda2/mu);
    
   % updata multipliers
  
    Y1 = Y1+ mu*(X-X*Z-E);
    Y2 = Y2+ mu*(Z-J);
    mu = min(pho*mu, max_mu);  %这句是什么意思
    
    % convergence conditions
    thrsh = 0.0001;
    if(norm(X-X*Z-E,inf)<thrsh && norm(Z-J,inf)<thrsh)
        IsConverge = 1;
    end
    cov_val(iter) = norm(X-X*Z-E,inf);

 if (iter==1 || mod(iter,50)==0 || IsConverge == 1)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-4*norm(Z,2))) ',cov_val(iter)' num2str(cov_val(iter),'%2.3e')]);
 end    
    iter = iter + 1;
end

[nmi,ACC,AR,f,p,r,RI]=clustering(abs(Z)+abs(Z'), cls_num, gt);