function drv = AcoustContrastControl(numSPK,Gb, Gq)
    regt = 100;
    if ndims(Gb) == 2
        tA = inv(conj(Gq).'*Gq + regt * eye(numSPK)) * conj(Gb).' * Gb;
        [Vx,~] = eig(tA);
        drv = Vx(:,1);
    else
        TPs = pagemtimes(permute(conj(Gq), [2 1 3]) , Gq);
        G_inv = TPs;
        Gtx = pagemtimes(permute(conj(Gb), [2 1 3]) , Gb);
        drv = zeros(numSPK, size(Gb,3));
        for ji = 1:size(Gb,3)
            st = TPs(:,:,ji);
            stg = Gtx(:,:,ji);
            G_inv(:,:,ji) = (inv(st + regt * eye(numSPK)))*stg;
            % G_inv(:,:,ji) = stg - 500 .* st; 
            tA = G_inv(:,:,ji);
            [Vx,~] = eig(tA);
            tdrv = Vx(:,1);
            drv(:,ji) = tdrv;
        end
    end
end
