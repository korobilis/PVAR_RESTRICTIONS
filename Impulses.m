clear all;
close all;
clc;

load('EA_PVAR_M1.mat');

names = {'AUSTRIA'; 'BELGIUM'; 'FINLAND'; 'FRANCE'; 'GREECE'; 'IRELAND'; 'ITALY'; 'NETHERLANDS'; 'PORTUGAL'; 'SPAIN'};
nhor = 12;
impulses_draws = zeros(length(thin),k*NG,nhor+1);
for irep = 1:length(thin)
    if mod(irep,250)==0
        disp(irep)
    end
    % impose the restrictions
    % 1) DI restrictions
    GAMMA = ones(n,1);
    for kk = 1:n_DI            
        ind_temp = index_restr_DI(:,:,kk)';
        GAMMA(ind_temp(:),1) = gammaDI_draws(irep,kk);
    end
    GAMMA = diag(GAMMA);

    % 3) SI restrictions
    for kk = 1:n_SI
        index_temp = index_restr_SI(:,:,kk);           
        GAMMA_SI(index_temp(:)) = gammaSI_draws(irep,kk);
        GAMMA_SI = reshape(GAMMA_SI,NG,NG);
    end
    sigma = GAMMA_SI.*squeeze(SIGMA_draws(irep,:,:));
    alpha_mat = reshape(alpha_draws(irep,:)*GAMMA,k,NG);
    A = [alpha_mat'; eye(NG*(p-1)) zeros(NG*(p-1),NG)];
    shock = chol(sigma)';
    shock = inv(diag(diag(shock)))*shock;
    IRF = IRFVAR(A,shock,p,nhor);
    impulses_draws(irep,:,:) = IRF;
end

IRF = squeeze(quantile(impulses_draws(:,5:NG:k*NG,:),[.16,.50,.84]));

save('IRFsaveM1.mat','IRF');

figure
set(gcf,'units','normalized','outerposition',[0 0 1 1]);
for i=1:10
    subplot(round(10/2),2,i)
    shadedplot(1:nhor,squeeze(cumsum(IRF(1,i,1:nhor)))',squeeze(cumsum(IRF(3,i,1:nhor)))','b','b')   
    hold all
    plot(squeeze(cumsum(IRF(2,i,1:nhor)))','black','Linewidth',2)
    xlim([1 10])
    if i==1
        ylim([-15 20])
    elseif i==2
        ylim([-5 20])
    elseif i==3
        ylim([-20 10])
    elseif i==4
        ylim([-20 20])
    elseif i==5
        ylim([-5 5])        
    elseif i==6
        ylim([-5 5])
    elseif i==8
        ylim([-15 15])
    elseif i==9
        ylim([-5 5])        
    elseif i==10
        ylim([-15 5])
    else
        ylim([-10 10])
    end
    grid on
    title(['Impulse responses of bond yield of ' cell2mat(names(i))])
end