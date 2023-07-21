%% G-minimization step
%
% shrinakge operator for weighted l1 norm
function G = shrinkage(V,W,gam,rho)
         a = (gam / rho) * W;         
         G = (V - a) .* ( V > a ) + (V + a) .* ( V < -a );
end