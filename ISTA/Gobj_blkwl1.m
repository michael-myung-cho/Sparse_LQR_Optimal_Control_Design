function [Gobj] = Gobj_blkwl1(V,W,blksize)
% Calulating G(K)

    Gobj = 0;
 
    mm = blksize(1);
    nn = blksize(2);
    
    sub_mat_size = size(V)./blksize;
    p = sub_mat_size(1);
    q = sub_mat_size(2);
    
    for i = 1:p
        for j = 1:q
            wij = W(i,j);            
            Vij = V( mm * (i-1) + 1:mm*i, nn * (j-1) + 1:nn*j);
            Gobj = Gobj+wij*norm( Vij, 'fro' );
        end
    end
            
end

