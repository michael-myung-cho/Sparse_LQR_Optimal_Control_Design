%% G-minimization step
%
% shrinakge operator for weighted l1 norm
function G = shrinkage_LearnedPara(V,W,LearnedPara)
         a = LearnedPara * W;         
         G = (V - a) .* ( V > a ) + (V + a) .* ( V < -a );
end