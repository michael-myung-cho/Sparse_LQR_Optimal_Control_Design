% Centralized solution with sparse feedback constraint
% GraSP in Greedy Sparsity-Constrained Optimization Bahmani, Sohail
% replace ComputeCentra.m with this
% used in paper for Algorithm 1


function result2 = GraSP_SparseK(A,B1,B2,Q,R,nnz_val)

sys_para.A = A;
sys_para.B2 = B2;
sys_para.B1 = B1;
sys_para.R = R;

% optimization parameters
options.tol = 10^(-4);
options.tolPolish = 1e-3;
options.outMax = 1000;
options.method = 'CG';
options.ADMMcomp = 0;
options.initChoice = 0;

K_WAC= lqr(sys_para.A,sys_para.B2,Q,sys_para.R);


%nnz_val = 100; % 


% another initial point: directly tuncating

%Kx = proj_l0(K_WAC,nnz_val);

%if max(real(eig(sys_para.A-sys_para.B2*Kx))) > 0
%    error('Directly truncating not stable\n');
%end

gen_blk = zeros(10,15);
result2 = GraSPutil_ComputeCentra(sys_para,Q,options, nnz_val,gen_blk);

end






