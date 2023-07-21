

load('Ksparse_x_f_y_dataset_normalized_1000.mat');

N=15;
M=5;

A_dataset = x_train_data_normalized(1:N*N,:);
B1_dataset =  x_train_data_normalized(N*N+1:N*N+N*N,:);
B2_dataset =  x_train_data_normalized(N*N+N*N+1:end,:);
F_dataset = Finit_train_data_normalized;

for ii=1:1000
    A=reshape(A_dataset(:,ii),N,N);
    B1=reshape(B1_dataset(:,ii),N,N);
    B2=reshape(B2_dataset(:,ii),N,M);
    F=reshape(F_dataset(:,ii),M,N);
    eig_vec = eig(A-B2*F);
    if isnan(eig_vec) | isinf(eig_vec)
        disp('error')
        break;
    end
end

