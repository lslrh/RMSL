function [W1, W2] = ext_updateNetwork(X,H,W1,W2,alpha,gamma,eta,max_iter,d1)
V = size(X,2); 
for v=1:V
    IsConverge = 0; iter = 0; I = eye(d1{v},d1{v});
    while (IsConverge == 0&&iter<max_iter+1)
        norm(X{v}-W2{v}*tanh(W1{v}*H),'fro');
        % update W2
        M{v} = tanh(W1{v}*H);
        W2{v} = X{v}*M{v}'/(M{v}*M{v}'+gamma/alpha{v}*I+I*1e-6);  
%        norm(X{v}-W2{v}*tanh(W1{v}*H),'fro')
        % update W1
        gad_W1{v} = alpha{v}*((1-M{v}.*M{v}).* (W2{v}'*W2{v}*M{v}-W2{v}'*X{v}))*H'+gamma*W1{v};      
        W1{v} = W1{v} - eta*gad_W1{v};
        % check convergence
        thrsh = 1e-5;
        if(norm(X{v}-W2{v}*tanh(W1{v}*H),'fro')<thrsh)
            IsConverge = 1;
        end
        norm(X{v}-W2{v}*tanh(W1{v}*H),'fro');
        iter = iter + 1;
    end
end