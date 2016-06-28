function grdecl_new = padGrdecl(grdecl,dirs,box,varargin)
    opt=struct('relative',false);
    opt=merge_options(opt,varargin{:});
    [~,zcorn]=grdeclXYZ(grdecl);
    if(dirs(3))
        top=box(3,1);bottom=box(3,2);
        if(top<min(0) || bottom<0)
            warning('Boxing z direction not possible')
        else
        cartdims=grdecl.cartDims+[0 0 2];
        zcorn_new=nan(cartdims*2);
        zcorn_new(:,:,3:end-2)=zcorn;
        zcorn_new(:,:,2)=zcorn(:,:,1);
        zcorn_new(:,:,1)=min(zcorn(:))-top;
        zcorn_new(:,:,end-1)=zcorn(:,:,end);
        zcorn_new(:,:,end)=max(zcorn(:))+bottom;
        grdecl_new=struct('cartDims',cartdims,'COORD',grdecl.COORD,'ZCORN',zcorn_new(:));
        end
    end
    if(~any(dirs(1:2)))
        return
    end
        %
        %%{
   % rotate model and move to "origo"   
        grdecl=grdecl_new;        
        [xyz,zcorn]=grdeclXYZ(grdecl);
        vecx=xyz([1 2],end,1)-xyz([1 2],1,1);
        vecy=xyz([1 2],1,end)-xyz([1 2],1,1);
        vecx=vecx/norm(vecx)
        vecy=vecy/norm(vecy)
        vecy=vecy-sum(vecx.*vecy,1)*vecx;
        vecy=vecy/norm(vecy)
        U=[vecx,vecy];
        xyz_ss=size(xyz);
        xyz([1,2],:,:)=reshape(U'*reshape(xyz([1,2],:,:),2,[]),2,xyz_ss(2),xyz_ss(3));
        xyz([4,5],:,:)=reshape(U'*reshape(xyz([4,5],:,:),2,[]),2,xyz_ss(2),xyz_ss(3));
        origo=min(reshape(xyz([1,2],:,:),2,[])')';
        xyz([1,2],:,:)=bsxfun(@minus,xyz([1,2],:,:),origo);
        xyz([4,5],:,:)=bsxfun(@minus,xyz([4,5],:,:),origo);
        org_max=max(reshape(xyz([1,2],:,:),2,[])')';
        org_min=min(reshape(xyz([1,2],:,:),2,[])')';
       
        %%{
        % pad i xretning
        if(dirs(1))
        cartdims=grdecl.cartDims+[2 0 0];
        zcorn_new=nan(cartdims*2);
        zcorn_new(3:end-2,:,:)=zcorn;
        zcorn_new(2,:,:)=zcorn(1,:,:);
        zcorn_new(1,:,:)=zcorn(1,:,:);
        zcorn_new(end-1,:,:)=zcorn(end,:,:);
        zcorn_new(end,:,:)=zcorn(end,:,:);
        xyz_ss=size(xyz);
        xyz_new=nan(xyz_ss+[0 2 0]);
        xyz_new(:,2:end-1,:)=xyz;
        xyz_new(:,1,:)=xyz(:,1,:);
        xyz_new(:,end,:)=xyz(:,end,:);
        if(opt.relative)
            vec=[1 1];
            pos1=[-box(1,1), 0];
            pos2=[box(2,2),  0];
        else
            vec=[0 1];
            pos1=[org_min(1)-box(1,1),0];
            pos2=[org_max(1)+box(1,2),0];
        end
        
        xyz_new([1,2],1,:)=bsxfun(@times,xyz_new([1,2],1,:),vec');
        xyz_new([1,2],1,:)=bsxfun(@plus,xyz_new([1,2],1,:),pos1');
        xyz_new([1,2],end,:)=bsxfun(@times,xyz_new([1,2],end,:),vec');
        xyz_new([1,2],end,:)=bsxfun(@plus,xyz_new([1,2],end,:),pos2');
        xyz_new([4,5],1,:)=xyz_new([1,2],1,:);
        xyz_new([4,5],end,:)=xyz_new([1,2],end,:);
        if(box(1,1)<org_min(1) && box(1,2)<org_max(1))
            warning('boxing with this box in x direction not possible');
        else
            grdecl_new=struct('cartDims',cartdims,'COORD',xyz_new(:),'ZCORN',zcorn_new(:));
        end
        end
        % pad y yretning
        if(dirs(2))
        [xyz,zcorn]=grdeclXYZ(grdecl_new);
        cartdims=grdecl_new.cartDims+[0 2 0];
        zcorn_new=nan(cartdims*2);
        zcorn_new(:,3:end-2,:)=zcorn;
        zcorn_new(:,2,:)=zcorn(:,1,:);
        zcorn_new(:,1,:)=zcorn(:,1,:);
        zcorn_new(:,end-1,:)=zcorn(:,end,:);
        zcorn_new(:,end,:)=zcorn(:,end,:);
        xyz_ss=size(xyz);
        xyz_new=nan(xyz_ss+[0 0 2]);
        xyz_new(:,:,2:end-1)=xyz;
        xyz_new(:,:,1)=xyz(:,:,1);
        xyz_new(:,:,end)=xyz(:,:,end);
        if(opt.relative)
            vec=[1 1];
            pos1=[0,-box(2,1)];
            pos2=[0 box(2,2)];
        else
            vec=[1 0];
            pos1=[0,org_min(2)-box(2,1)];
            pos2=[0 org_max(2)+box(2,2)];
        end
        xyz_new([1,2],:,1)=bsxfun(@times,xyz_new([1,2],:,1),vec');
        xyz_new([1,2],:,1)=bsxfun(@plus,xyz_new([1,2],:,1),pos1');
        xyz_new([1,2],:,end)=bsxfun(@times,xyz_new([1,2],:,end),vec');
        xyz_new([1,2],:,end)=bsxfun(@plus,xyz_new([1,2],:,end),pos2');
        
        xyz_new([4,5],:,1)=xyz_new([1,2],:,1);
        xyz_new([4,5],:,end)=xyz_new([1,2],:,end);
        if(box(1,1)<org_min(1) && box(1,2)<org_max(1))
            warning('boxing with this box in x direction not possible');
        else
            grdecl_new=struct('cartDims',cartdims,'COORD',xyz_new(:),'ZCORN',zcorn_new(:));
        end
        end
end