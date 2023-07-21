%% Multi-agent control fine tuning scenario
% Discription: Suppose we have a nominal system model. By learning
% DNN-ISTA/FISTA offline, we can obtain approximate sparse controller.
% After that, when we get to have a specific system model, using the
% approximate sparse controller as a warm-start of ISTA algorithm for
% fine-tune.
%
% By Myung (Michael) Cho
% 06/21/2023


clear;
clc;

addpath('02_ISTA_with_learnedPara');
addpath('03_ISTA_with_warmStart');
addpath('../../ISTA');
%% State-space representation (Nominal system model)
% number of systems
N = 5;

% size of each system
nn = 3; 
mm = 1;

% use cyclic condition to obtain unstable system
a = ones(1,nn);
b = 1.5*(sec(pi/nn))*a;

% state-space representation of each system
Aa = -diag(a) + diag(b(2:nn),-1); 
Aa(1,nn) = -b(1);
Bb1 = diag(b);
Bb2 = zeros(nn,2); % MCho
Bb2(1) = b(1);      
Bb2(2,2) = b(1); % MCho   

% non-symmetric weighted Laplacian matrix 

% adjacency matrix 
Ad = toeplitz([1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]);
for i = 1 : N
    for j = 1 : N
        if i ~= j
            cij = 0.5 * ( i - j );
        else
            cij = 0;
        end
        Ad( nn*(i-1)+1 : nn*i, nn*(j-1)+1 : nn*j) = cij * eye(nn);
    end
end
        
% take the sum of each row
d = sum(Ad,2);

% form the Laplacian matrix
L = Ad - diag(d);

% state-space representation of the interconnected system

A  = kron(eye(N), Aa) - L;
B1 = kron(eye(N), Bb1);
B2 = kron(eye(N), Bb2);
Q  = eye(nn*N);
R  = eye((nn-1)*N);


%% statisical result
Ntrial=100;
AvgTime_ISTA=0;
AvgNumItr_ISTA=0;
AvgJ_ISTA=0;
AvgNNZ_ISTA=0;
AvgTime_ISTA_warmStart=0;
AvgNumItr_ITSA_warmStart=0;
AvgJ_ISTA_warmStart=0;
AvgNNZ_ISTA_warmStart=0;

for ii=1:Ntrial

%% A special model (as an example of fine tuning by adding a noise)
A_nor = A;
B1_nor = B1;
B2_nor = B2;
Noise=(abs(A_nor)>10^-6).*(randn(N*nn,N*nn)*0.1);
A_nor_noise = A_nor+Noise;

%% Generating a warm-start Fhat with learned parameter from DNN-ISTA 

% Reading learned parameters in NPY file (obtained from python)
conversion_npz_to_mat;  % Run to get learned parameters in matlab format, LearnedPara variable has the information.

Fhat_warmstart = ISTA_SparseK_LearnedPara(A_nor_noise,B1_nor,B2_nor,Q,R,LearnedPara);   % DNN-ISTA mimic


%% Compute the optimal sparse feedback gains
options = struct('method','l1','gamval', 1,'rho',100,'maxiter',10000,'blksize',[1 3],'reweightedIter',1);


Solpath_ISTA=ISTA_SparseK(A_nor_noise,B1_nor,B2_nor,Q,R,options);
Solpath_ISTA_warmStart=ISTA_SparseK_warmStart(A_nor_noise,B1_nor,B2_nor,Q,R,Fhat_warmstart,options);



%% calculating statistical result
AvgTime_ISTA=AvgTime_ISTA+Solpath_ISTA.tBuf;
AvgNumItr_ISTA=AvgNumItr_ISTA+Solpath_ISTA.itrBuf;
AvgJ_ISTA=AvgJ_ISTA+Solpath_ISTA.J;
AvgNNZ_ISTA=AvgNNZ_ISTA+Solpath_ISTA.nnz;

AvgTime_ISTA_warmStart=AvgTime_ISTA_warmStart+Solpath_ISTA_warmStart.tBuf;
AvgNumItr_ITSA_warmStart=AvgNumItr_ITSA_warmStart+Solpath_ISTA_warmStart.itrBuf;
AvgJ_ISTA_warmStart=AvgJ_ISTA_warmStart+Solpath_ISTA_warmStart.J;
AvgNNZ_ISTA_warmStart=AvgNNZ_ISTA_warmStart+Solpath_ISTA_warmStart.nnz;

end
AvgTime_ISTA=AvgTime_ISTA/Ntrial
AvgTime_ISTA_warmStart=AvgTime_ISTA_warmStart/Ntrial

AvgNumItr_ISTA=AvgNumItr_ISTA/Ntrial
AvgNumItr_ITSA_warmStart=AvgNumItr_ITSA_warmStart/Ntrial

AvgJ_ISTA=AvgJ_ISTA/Ntrial
AvgJ_ISTA_warmStart=AvgJ_ISTA_warmStart/Ntrial


AvgNNZ_ISTA=AvgNNZ_ISTA/Ntrial
AvgNNZ_ISTA_warmStart=AvgNNZ_ISTA_warmStart/Ntrial





