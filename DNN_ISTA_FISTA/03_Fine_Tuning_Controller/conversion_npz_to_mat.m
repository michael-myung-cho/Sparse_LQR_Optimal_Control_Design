%% conversion_npz_to_mat
% after extracting npz file to npy files, run this file to read learned
% parameters including stepsize and thresholding level in DNN-ISTA
%
% By Myung (Michael) Cho
% 06/021/2023

%% Output
% LearnedPara: matrix (# of layers) x 2 
% 1st column: learned stepsize
% 2nd column: learned threshoding level 
% Number of rows: Number of layers in DNN

%%

addpath('01_npy_to_matlab/npy-matlab');

PathName='00_Learned_Parameter/Ksparse_x_f_y_dataset_N5_1000_T30/layer';

T=30;
LearnedPara=[];
for ii=0:T-1
    FolderName=strcat(PathName,num2str(ii),'/');
    FileName1=strcat(FolderName,'rhoStep_',num2str(ii),':0.npy');
    FileName2=strcat(FolderName,'rhoThre_',num2str(ii),':0.npy');
    LearnedPara=[LearnedPara;readNPY(FileName1),readNPY(FileName2)];
end
system('cd ..');
system('cd ..');
system('cd ..');