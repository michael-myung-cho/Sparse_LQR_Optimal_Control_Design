%% Iterative Soft-thresholding Algo. for sparse K prediction with learned parameter
% By Myung (Michael) Cho
%
% ----

function Fhat = ISTA_SparseK_LearnedPara(A,B1,B2,Q,R,LearnedPara)


%% absolute tolerances for the stopping criterion
NumItr = size(LearnedPara,1);    % number of layers in DNN

%% Initial LQR point
F = lqr(A,B2,Q,R);

%% define funtion

max_eig = @(F) max(real(eig(A-B2*F)));
lyap_P = @(F) lyap((A - B2 * F)', Q + F' * R * F);
lyap_L = @(F) lyap((A - B2 * F), B1*B1');
gradJ_fun = @(F,P,L) 2*(R*F - B2'*P)*L;

%% Solve the minimization problem using Iterative Soft-Thresholding
for itr=1:NumItr

    F_pre = F;

    P = lyap_P(F_pre);
    L = lyap_L(F_pre);
    gradJ = gradJ_fun(F_pre, P, L);

    % gradient descent
    F = F_pre - gradJ*LearnedPara(itr,1);

    % shrinkage operator
    F = shrinkage_LearnedPara(F,ones(size(F)),LearnedPara(itr,2));

    if (max_eig(F) >= 0)
        F = F_pre;      % bypass: mimic DNN-ISTA
    end


    %% display (for debugging)
    quiet=1;
    if ~quiet
        fprintf('%5d \t %6.1f \t %6.3f \t %6.3f \t %6.3f \n', itr, gam, resF, norm(gradJ,'fro'), Fobj(F,gam));
    end

end
Fhat = F;   % Sparse controller with learned parameter 


end