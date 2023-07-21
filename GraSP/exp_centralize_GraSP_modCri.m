% Centralized solution with sparse feedback constraint
% GraSP in Greedy Sparsity-Constrained Optimization Bahmani, Sohail
% replace ComputeCentra.m with this
% used in paper for Algorithm 1


function result2 = GraSP_SparseK(A,B1,B2,Q,R)

sys_para.A = A;
sys_para.B2 = B2;
sys_para.B1 = B1;
sys_para.R = R;

% optimization parameters
options.tol = 10^(-5);
options.tolPolish = 1e-4;
options.outMax = 100;
options.method = 'CG';
options.ADMMcomp = 0;
options.initChoice = 0;

K_WAC= lqr(sys_para.A,sys_para.B,Q,sys_para.R);


nnz_val = 100; % 


% another initial point: directly tuncating

Kx = proj_l0(K_WAC,nnz_val);

if max(real(eig(sys_para.A-sys_para.B*Kx))) > 0
    error('Directly truncating not stable\n');
end


result2 = GraSPutil_ComputeCentra(sys_para,Q,options, nnz_val,Kx);

end

function F = proj_l0(F_temp,nnz_val)
    [m,n]=size(F_temp);
    [~,ind] = sort(abs(F(:)),'descend');
    F=zeros(m,n);
    F(ind(1:nnz_val))=F_temp(ind(1:nnz_val));
end





