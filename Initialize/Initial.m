%% LRR
load('ORL_mtv.mat');
N = size(X{1},2); 
fprintf('Latent representation multiview subspace clustering\n');
num_views = size(X,2);
numClust = size(unique(gt),1);
Z = cell(1,num_views);
lambda2 = 1;

for j=1:num_views
          [nmi,ACC,AR,f,p,r,RI,Z{j}] = concateLRR(X,gt,numClust,1,lambda2,j);
          result=[nmi,ACC,f,RI];            
end

save(strcat('C:\Users\¿Ó»ÔªÕ\Desktop\RMSL\RMSL\Z',filesep,'ORL',...
  '-lambda2=',num2str(lambda2),'.mat'),'Z');
