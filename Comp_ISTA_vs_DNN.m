clear; clc;
load('./DNN_ISTA_FISTA/02_DNN_FISTA/Ksparse_x_f_y_dataset_N5_1000.mat');
addpath('ISTA');

Num_test=100;
NMSE_10=0;
NMSE_20=0;
NMSE_30=0;
NMSE_50=0;
NMSE_100=0;
NMSE_200=0;
NMSE_300=0;

for ii=1:Num_test
    ii
    
    A_vec=x_test_data(ii,1:15*15);
    B1_vec=x_test_data(ii,15*15+1:15*15+15*15);
    B2_vec=x_test_data(ii,15*15+15*15+1:end);
    
    A=reshape(A_vec,15,15);
    B1=reshape(B1_vec,15,15);
    B2=reshape(B2_vec,15,10);
    
    %%%%%%%% ISTA %%%%%%
    fprintf('ISTA with 10 iter \n');
    options_ista = struct('method','l1','gamval', 1,'rho',100,'maxiter',10,'blksize',[1 3],'reweightedIter',1);
    solpath_ista_10 = ISTA_SparseK(A,B1,B2,Q,R,options_ista);
    
    NMSE_10 = NMSE_10 + (norm(solpath_ista_10.F(:)' - y_test_data(ii,:))^2)/(norm(y_test_data(ii,:))^2);
     
    %%%%%%%% ISTA %%%%%%
    fprintf('ISTA with 20 iter \n');
    options_ista = struct('method','l1','gamval', 1,'rho',100,'maxiter',20,'blksize',[1 3],'reweightedIter',1);
    solpath_ista_20 = ISTA_SparseK(A,B1,B2,Q,R,options_ista);

    NMSE_20 = NMSE_20 + norm(solpath_ista_20.F(:)' - y_test_data(ii,:),2)^2/norm(y_test_data(ii,:),2)^2;

    %%%%%%%% ISTA %%%%%%
    fprintf('ISTA with 30 iter \n');
    options_ista = struct('method','l1','gamval', 1,'rho',100,'maxiter',30,'blksize',[1 3],'reweightedIter',1);
    solpath_ista_30 = ISTA_SparseK(A,B1,B2,Q,R,options_ista);

    NMSE_30 = NMSE_30 + norm(solpath_ista_30.F(:)' - y_test_data(ii,:),2)^2/norm(y_test_data(ii,:),2)^2;
    
    %%%%%%%% ISTA %%%%%%
    fprintf('ISTA with 50 iter \n');
    options_ista = struct('method','l1','gamval', 1,'rho',100,'maxiter',50,'blksize',[1 3],'reweightedIter',1);
    solpath_ista_50 = ISTA_SparseK(A,B1,B2,Q,R,options_ista);

    NMSE_50 = NMSE_50 + norm(solpath_ista_50.F(:)' - y_test_data(ii,:),2)^2/norm(y_test_data(ii,:),2)^2;
    
    %%%%%%%% ISTA %%%%%%
    fprintf('ISTA with 100 iter \n');
    options_ista = struct('method','l1','gamval', 1,'rho',100,'maxiter',100,'blksize',[1 3],'reweightedIter',1);
    solpath_ista_100 = ISTA_SparseK(A,B1,B2,Q,R,options_ista);

    NMSE_100 = NMSE_100 + norm(solpath_ista_100.F(:)' - y_test_data(ii,:),2)^2/norm(y_test_data(ii,:),2)^2;
    
    %%%%%%%% ISTA %%%%%%
    fprintf('ISTA with 200 iter \n');
    options_ista = struct('method','l1','gamval', 1,'rho',100,'maxiter',200,'blksize',[1 3],'reweightedIter',1);
    solpath_ista_200 = ISTA_SparseK(A,B1,B2,Q,R,options_ista);

    NMSE_200 = NMSE_200 + norm(solpath_ista_200.F(:)' - y_test_data(ii,:),2)^2/norm(y_test_data(ii,:),2)^2;    
    
    %%%%%%%% ISTA %%%%%%
    fprintf('ISTA with 300 iter \n');
    options_ista = struct('method','l1','gamval', 1,'rho',100,'maxiter',300,'blksize',[1 3],'reweightedIter',1);
    solpath_ista_300 = ISTA_SparseK(A,B1,B2,Q,R,options_ista);

    NMSE_300 = NMSE_300 + norm(solpath_ista_300.F(:)' - y_test_data(ii,:),2)^2/norm(y_test_data(ii,:),2)^2;    

end

NMSE_10=NMSE_10/Num_test
NMSE_20=NMSE_20/Num_test
NMSE_30=NMSE_30/Num_test
NMSE_50=NMSE_50/Num_test
NMSE_100=NMSE_100/Num_test
NMSE_200=NMSE_200/Num_test
NMSE_300=NMSE_300/Num_test

