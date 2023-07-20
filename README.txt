---- Written by Myung (Michael) Cho
---- 06/21/2023

------------Title-----------
Iterative Thresholding and Projection Algorithms and Model-Based Deep Neural Networks for Sparse LQR Control Design


-----------Description----------
Algorithm implementation for sparse LQR optimal control design

- Iterative Soft-thresholding Algorithm (ISTA) for sparse LQR optimal control design
- Iterative Sparse Projection Algorithm (ISPA) for sparse LQR optimal control design
- Learning based ISTA (DNN-ISTA)
- Learning based Fast ISTA (DNN-FISTA)


------------Folders----------------
ADMM: Alternating Direction Method of Multiplier (Previous work by Fu Lin, et. al.)
GraSP: Gradient Support Pursuit (Previous work by Feier Lian, et. al.)

ISTA: Iterative Soft-thresholding Algorithm for sparse LQR (Matlab used)
ISPA: Iterative Sparse Projection Algorithm for sparse LQR (Matlab used)
DNN_ISTA_FISTA: DNN-ISTA, DNN_FISTA (python & tensorflow used to implement DNNs)


------------Files to run (on Matlab)----------------
Test_multiagent.m: Distributed multi-agent control system (Introduced in Section VII)
Test_multiagent2.m: Another distributed multi-agent control system (Introduced in Section VII)
Test_spatially_dist.m: Another example of distributed multi-agent control system
Comp_ISTA_vs_DNN.m: Calculating Normalized Mean Square Error (NMSE) for comparison between DNN-ISTA/FISTA and ISTA


------------References-----------
[1] Myung Cho and Aranya Chakrabortty, "Iterative shrinkage-thresholding algorithm and model-based neural network for sparse LQR control design," in Proceedings of European Control Conference (ECC), 2022, pp. 2311– 2316. 
[2] Myung Cho, "Deep Neural Networks Based on Iterative Thresholding and Projection Algorithms for Sparse LQR Control Design," arXiv preprint arXiv:2212.02929.
