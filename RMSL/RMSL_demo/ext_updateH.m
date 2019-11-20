function [H] = ext_updateH(X,H,Z,W1,W2,alpha,eta,max_iter)
V = size(X,2);
N = size(Z,1);
IsConverge = 0; iter = 0;
I = eye(N,N);
while (IsConverge == 0&&iter<max_iter+1)
    % calculate the gradient of H
    old_fv = 0;
    for v=1:V
       % old_fv =old_fv +  alpha{v}*norm(X{v}-W2{v}*tanh(W1{v}*H),'fro')+mu/2*norm(H-H*Z-Ec,'fro')+trace(Y2'*(H-H*Z-Ec));
        old_fv =old_fv + alpha{v}*norm(X{v}-W2{v}*tanh(W1{v}*H),'fro')+ norm(H-H*Z,'fro');  
    end
    old_fv;
    TM = 0;
    for v=1:V
        M{v} = tanh(W1{v}*H);
        TM = TM + alpha{v}*W1{v}'*((1-M{v}.*M{v}).* (W2{v}'*W2{v}*M{v}-W2{v}'*X{v}));
    end

   grad_H = TM + H*(I-Z-Z'+Z*Z');
  % grad_H = TM+mu*(H-H*Z-Ec)*(I-Z)'+Y2*(I-Z)';
    H = H - eta*grad_H;
    
    new_fv = 0;
    for v=1:V
       % new_fv =new_fv + alpha{v}*norm(X{v}-W2{v}*tanh(W1{v}*H),'fro')+ mu/2*norm(H-H*Z-Ec,'fro')+trace(Y2'*(H-H*Z-Ec));
        new_fv =new_fv + alpha{v}*norm(X{v}-W2{v}*tanh(W1{v}*H),'fro')+ norm(H-H*Z,'fro');
    end
    new_fv;
    % check convergence
    thrsh = 1e-5;
    for v=1:V
        if(old_fv<new_fv)
            IsConverge = 1;
        end
    end
    iter = iter + 1;
end