% MonteCarlo_PVAR_BMA.m: Panel VAR model with heterogeneity restriction search using
%             Bayesian model averaging. This is the version that replicates
%             the Monte Carlo exercise.
%             This code replicates the results in Koop and Korobilis (2014)
% -------------------------------------------------------------------------
% We estimate an unrestricted VAR in SURE form:
%
%             Y[t] = alpha x X[t] + e[t],  
%
% where Y[t] = (y[11,t]', y[12,t]',...,y[NG,t]')' are G macro variables for
% N countries observed at time t, t=1,...,T;  X = I kronecker (Y[t-1],...,Y[t-p]); 
% alpha is a column vector of VAR coefficients;  e[t] ~ N(0, SIGMA) with SIGMA
% a covariance matrix.
% -------------------------------------------------------------------------
% Notes: The code does not allow an intercept (just demean the data Y[t])
%        This version works only with p=1 lags, for more lags additional
%        for loops need to be added which search the DI and CS restriction
%        matrices.
%
% Written by Dimitris Korobilis,
% University of Glasgow,
% March 2014
% -------------------------------------------------------------------------

clear all; close all; clc;

% Add path of random number generators
addpath('functions')
addpath('data')

gamma_DI_MC = [];
gamma_CS_MC = [];
gamma_SI_MC = [];

alpha_MCMC_MC = [];
sigma_MCMC_MC = [];

alpha_OLS_MC = [];
sigma_OLS_MC = [];

for nMC = 1:5
    nMC

%---------------------------| USER INPUT |---------------------------------
% Gibbs-related preliminaries
nsave = 3000;           % Number of draws to save
nburn = 2000;           % Number of draws to discard
ntot = nsave + nburn;  % Number of total draws
iter = 1000;            % Print every "iter" iteration

% VAR specification
p = 1;    % Number of lags
N = 3;    % Number of cross sections
G = 2;    % Number of VAR variables for each cross-section

% Restrictions to examine
restr_DI = 1;  % Dynamic Interdependencies
restr_CS = 1;  % Cross-Sectional Heterogeneity
restr_SI = 1;  % Static Interdependencies
%----------------------------| END INPUT |---------------------------------

% Load data
[Yraw,PHI] = simpvardgp();

% load spreads.dat;
% load ip.dat;
% load debt.dat;
% load bid_ask.dat;
% spreads = spreads(1:end-1,:);
% 
% Yraw = [spreads(:,1:5),ip(:,1:5)];%,debt(:,1:5)];

% Create VAR data matrices
[Traw, NG] = size(Yraw);
if NG ~= N*G; error('wrong specification of N and G'); end  % Check dimensions
Ylag = mlag2(Yraw,p);
n = p*NG*NG;          % total number of regression coefficients
k = p*NG;             % number of coefficients in each equation
X = Ylag(p+1:Traw,:); % VAR data (RHS) matrix on the original specification    
x = kron(eye(NG),X);  % VAR data (RHS) matrix on the SURE model
% Correct time-series dimension of Y due to taking lags
Y = Yraw(p+1:Traw,:); 
T=Traw-p;
y = Y(:);    % This is the final matrix y = vec()Y

[x_t,~] = create_RHS_noint(Yraw,NG,p,Traw);
y_t = Y;
yy = y_t'; yy = yy(:);

% ====| Examine restrictions
% All VAR coefficients are in a single NG x NG x p column vector. With the
% code below I am trying to index the groups of elements of this huge vector 
% which correspond to C-S Heterogeneity, and DI restrictions. 
index_restriction = zeros(G,G,N*N);
index_var=zeros(G,G,N);   
for i_country = 1:N           
    index_temp = (i_country-1)*G+1:i_country*G;
    for i_variable = 1:G
        index_var(i_variable,:,i_country) = index_temp + (i_variable-1)*NG;
    end
end
% Note that for p=1 lags we have a single NGxNG coefficient matrix and an NGxNG covariance matrix, 
% so the index_restriction variable can be used to obtain both DI and CS restrictions, as well as SI restrictions
for i_country = 1:N
    index_restriction(:,:,(i_country-1)*N+1:i_country*N) = index_var + (i_country-1)*N*G*G;
end

% Now take indexes of the position of each restriction.
% Position of C-S heterogeneity restrictions
CS_index = 1:N+1:N*N;
% Position of DI interdependency restrictions
DI_index = 1:N*N; 
DI_index(CS_index) = [];   
% Position of SI interdependency restrictions
SI_index=[];
for kk = 2:N
    for nn = 1:kk-1
        SI_index = [SI_index; kk + N*(nn-1)]; %#ok<AGROW>
    end
end
SI_index = sort(SI_index); % Sort the indexes from smallest to largest number 

n_CS = N*(N-1)/2;    % Number of C-S restrictions
n_DI = length(DI_index);   % Number of DI restrictions
n_SI = length(SI_index);   % Number of DI restrictions

index_CS = index_restriction(:,:,CS_index); % To be used to obtain index of C-S restrictions
index_restr_DI = index_restriction(:,:,DI_index); % Index of DI restrictions
index_restr_SI = index_restriction(:,:,SI_index); % Index of DI restrictions

% The "index_CS" variable indexes the matrices A_{1}^{i} of country i.
% We need to test pairs of restrictions of the form A_{1}^{i} = A_{1}^{j}
% in order to test homogeneity of countries i and j.
% a) First create pairs of countries:
pairs_index = combntns(1:N,2);   % Index of pairs
% b) Second obtain index of CS restrictions. For each pair of countries, we
% are testing the equivalence of the GxG matrices A_{1}^{i}.
index_restr_CS = cell(n_CS,1);
index_rest_CS_all = zeros(G*G*n_CS,2);
for ir = 1:n_CS
    temp =  index_CS(:,:,pairs_index(ir,:));
    temp1 = temp(:,:,1); temp2 = temp(:,:,2);
    index_restr_CS{ir,1} =  [temp1(:),temp2(:)];
    index_rest_CS_all((ir-1)*G*G+1:ir*G*G,:) = [temp1(:),temp2(:)];
end

% ====| Set priors
% OLS quantities
alpha_OLS_vec = inv(x'*x)*(x'*y);
alpha_OLS_vec2 = inv(x_t'*x_t)*(x_t'*yy);
alpha_OLS_mat = inv(X'*X)*(X'*Y);
SSE = (Y - X*alpha_OLS_mat)'*(Y - X*alpha_OLS_mat);
sigma_OLS = SSE./(T-(k-1));
sigma = sigma_OLS;
sigma_inv = inv(sigma);
PSI_ALL = chol(inv(sigma))';

alpha = alpha_OLS_vec;
alpha_CS = cell(p,1);

% alpha ~ N(0,D^-1), where D^-1 = diag(tau2)
tau2 = zeros(n_DI,1);
ksi2 = zeros(n_CS,1);
kappa2 = zeros(n_SI,1);
c1 = 1e-6;
c2 = 1e-4;
c3 = 1e-5;

% tau^-2 ~ Gamma(rho1,rho2);
rho1 = 1;
rho2 = 50;

% ksi^-2 ~ Gamma(miu1,miu2);
miu1 = 5;
miu2 = 50;

% kappa2 ~ Gamma(lambda1,lambda2);
lambda1 = 1;
lambda2 = 25;

% (psi)^2 ~ GAMMA(a_i,b_i)
a_i = .01;
b_i = .01;

% gamma_j ~ Bernoulli(1,p_j)
p_j_DI = 0.5*ones(n_DI,1);
p_j_CS = 0.5*ones(n_CS,1);
p_j_SI = 0.5*ones(n_SI,1);

% Initialize parameters
gamma_DI = ones(n_DI,1);
gamma_CS = ones(n_CS,1);
gamma_SI = ones(n_SI,1);
GAMMA2 = cell(n_CS,1);
for i = 1:n_CS; GAMMA2{i,1} = speye(n); end
GAMMA = speye(n);
for i = 1:n_CS
    GAMMA = GAMMA*GAMMA2{i,1};
end
GAMMA_SI = ones(NG*NG,1);
VAR_SI = 2*ones(NG*NG,1);
var_param = ones(NG,NG);
S=cell(1,NG);
s=cell(1,NG-1);
omega=cell(1,NG-1);
for kk_1 = 1:(NG-1)
    omega{kk_1} = ones(kk_1,1);	% Omega_j
end
h=cell(1,NG-1);
for kk = 1:NG-1
   h{1,kk} = 2*ones(kk,1);
end
D_j=cell(1,NG-1);
R_j = cell(1,NG-1);
DRD_j=cell(1,NG-1);
B=cell(1,NG);
eta = cell(1,NG-1); 
psi_ii_sq = zeros(NG,1);

% Create storage matrices for posteriors
alpha_draws = zeros(nsave,n);
sigma_draws = zeros(nsave,NG,NG);
gammaDI_draws = zeros(nsave,n_DI);
gammaCS_draws = zeros(nsave,n_CS);
gammaSI_draws = zeros(nsave,n_SI);

tic;
% ===============| GIBBS SAMPLER
for irep = 1:ntot
    % Print every "iter" iterations on the screen
    if mod(irep,iter)==0
        disp(irep)
        toc;
    end
   
    %------------------------------------------------------
    % STEP 1: Update VAR coefficients alpha from Normal
    %------------------------------------------------------
    h_i = (16)*ones(n,1);
    for kk = 1:n_DI
        ind_temp = index_restr_DI(:,:,kk)';
        % sample tau^-2 from Gamma
        r1 = rho1 + .5*G;
        r2 = rho2 + sum(alpha(ind_temp(:)).^2)/(2*(c1^(1-gamma_DI(kk,1))));
        tau2(kk,1) = 1./gamrnd(r1,1./r2);
        if gamma_DI(kk,1) == 0
           h_i(ind_temp(:)) = c1*tau2(kk,1);
        elseif gamma_DI(kk,1) == 1
           h_i(ind_temp(:)) = tau2(kk,1);
        end
    end
    
    for kk = 1:n_CS
        ind_temp = index_restr_CS{kk,1};
        % Sample ksi^-2 from Gamma
        r1 = miu1 + .5*G;
        r2 = miu2 + sum( (alpha(ind_temp(:,1))-alpha(ind_temp(:,2))).^2)/(2*(c2^(1-gamma_CS(kk,1))));
        ksi2(kk,1) = 1./gamrnd(r1,1./r2);        
        if gamma_CS(kk,1) == 0
           h_i(ind_temp(:,1)) = c2*ksi2(kk,1);
        elseif gamma_CS(kk,1) == 1
           h_i(ind_temp(:,1)) = ksi2(kk,1);
        end       
    end
    
    D = diag(1./h_i);
    psi_xx = kron((PSI_ALL*PSI_ALL'),(X'*X));
    Delta_alpha = inv(psi_xx + D);
    miu_alpha = Delta_alpha*((psi_xx)*alpha_OLS_vec);    
    alpha = GAMMA*miu_alpha + chol(Delta_alpha)'*randn(n,1);
%     alpha = GAMMA*alpha;
    alpha_mat = reshape(alpha,k,NG);
    
    %----------------------------------------------------------------------
    % STEP 2: Update DI and CS restriction indexes of alpha from Bernoulli
    %----------------------------------------------------------------------    
    if restr_DI == 1
        p_j_DI = repmat(betarnd(1 + sum(gamma_DI==1),1 + sum(gamma_DI~=1)),n_DI,1);
        for kk = 1:n_DI            
            ind_temp = index_restr_DI(:,:,kk)';
            u_i1 = mvnpdf(alpha(ind_temp(:)),zeros(G*G,1),c1*tau2(kk,1)*eye(G*G))*p_j_DI(kk);
            u_i2 = mvnpdf(alpha(ind_temp(:)),zeros(G*G,1),tau2(kk,1)*eye(G*G))*(1- p_j_DI(kk));
            gst = u_i2./(u_i1 + u_i2);
            gamma_DI(kk,1) = bernoullirnd(gst);   
        end
    end
    
    % 2) Examine cross-sectional (CS) heterogeneities
    if restr_CS == 1
        p_j_CS = repmat(betarnd(1 + sum(gamma_CS==1),1 + sum(gamma_CS~=1)),n_CS,1);
        for kk = 1:n_CS
            ind_temp = index_restr_CS{kk,1};           
            % Sample gamma_CS from Bernoulli
            u_i1 = mvnpdf(alpha(ind_temp(:,1)),alpha(ind_temp(:,2)),c2*ksi2(kk,1)*eye(G*G))*p_j_CS(kk);
            u_i2 = mvnpdf(alpha(ind_temp(:,1)),alpha(ind_temp(:,2)),ksi2(kk,1)*eye(G*G))*(1- p_j_CS(kk));
            gst = u_i2./(u_i1 + u_i2);
            gamma_CS(kk,1) = bernoullirnd(gst);
            if gamma_CS(kk) == 0
                for d_G = 1:G*G   
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,1)) = 0;
                    GAMMA2{kk,1}(ind_temp(d_G,2),ind_temp(d_G,2)) = 1;                   
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,2)) = 1;
                end
            else
                for d_G = 1:G*G
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,1)) = 1;
                    GAMMA2{kk,1}(ind_temp(d_G,2),ind_temp(d_G,2)) = 1;
                    GAMMA2{kk,1}(ind_temp(d_G,1),ind_temp(d_G,2)) = 0;
                end
            end
            GAMMA = speye(n);
            for i = 1:n_CS
                GAMMA = GAMMA*GAMMA2{i,1};
            end
        end
    end
    
    %------------------------------------------------------------------
    % STEP 3: Update VAR covariance matrix and SI restriction indexes
    %------------------------------------------------------------------
    SSE       = (Y - X*alpha_mat)'*(Y - X*alpha_mat);
    
    % Get S_[j] - upper-left [j x j] submatrices of SSE
    % The following loop creates a cell array with elements S_1,
    % S_2,...,S_j with respective dimensions 1x1, 2x2,...,jxj
    for kk_2 = 1:NG                         
        S{kk_2} = SSE(1:kk_2,1:kk_2);
    end

    % Set also SSE =(s_[i,j]) & get vectors s_[j]=(s_[1,j] , ... , s_[j-1,j])
    for kk_3 = 2:NG
        s{kk_3 - 1} = SSE(1:(kk_3 - 1),kk_3);
    end
    
    % Parameters for Heta|omega ~ N_[j-1](0,D_[j]*R_[j]*D_[j]), see eq. (15)
    % Create and update h_[j] matrix
    % If omega_[ij] = 0 => h_[ij] = kappa0, else...
    for kk_4 = 1:NG-1
        h{kk_4} = var_param(kk_4+1,1:kk_4);
    end

    % D_j = diag(h_[1j],...,h[j-1,j])
    for kk_5 = 1:NG-1
        D_j{kk_5} = diag(cell2mat(h(kk_5)));
    end

    % Now create covariance matrix D_[j]*R_[j]*D_[j], see eq. (15)
    for kk_6 = 1:NG-1
        DD = cell2mat(D_j(kk_6));
        DRD_j{kk_6} = (DD*DD);
    end

    % Create B_[i] matrix
    for rr = 1:NG
        if rr == 1
            B{rr} = b_i + 0.5*(SSE(rr,rr));
        elseif rr > 1
            s_i = cell2mat(s(rr-1));
            S_i = cell2mat(S(rr-1));
            DiRiDi = cell2mat(DRD_j(rr-1));
            B{rr} = b_i + 0.5*(SSE(rr,rr) - s_i'/(S_i + inv(DiRiDi))*s_i);
        end
    end

    % Now get B_i from cell array B, and generate (psi_[ii])^2
    B_i = cell2mat(B);
    for kk_7 = 1:NG
	    psi_ii_sq(kk_7,1) = gamm_rnd(1,1,(a_i + 0.5*T),B_i(1,kk_7));
    end
    
    ETA = zeros(NG,NG);
    for kk_8 = 1:NG-1
        s_i = cell2mat(s(kk_8));
        S_i = cell2mat(S(kk_8));
        DiRiDi = cell2mat(DRD_j(kk_8));
        miu_j = - sqrt(psi_ii_sq(kk_8+1))*((S_i + inv(DiRiDi))\s_i);
        Delta_j = inv(S_i + inv(DiRiDi));    
        eta{kk_8} = miu_j + chol(Delta_j)'*randn(kk_8,1);
        ETA(kk_8+1,1:kk_8) = eta{kk_8}';  
    end
    ETA_vec = ETA(:);
    
    if restr_SI == 1
        p_j_SI = repmat(betarnd(1 + sum(gamma_SI==1),1 + sum(gamma_SI~=1)),n_SI,1);
        for kk = 1:n_SI
            index_temp = index_restr_SI(:,:,kk);
            % First sample the kappas
            r1 = lambda1 + .5*G;
            r2 = lambda2 + sum(alpha(index_temp(:)).^2)/(2*(c3^(1-gamma_SI(kk,1))));
            kappa2(kk,1) = 1./gamrnd(r1,1./r2);
            % Then sample the gammas
            u_ij1 = mvnpdf(ETA_vec(index_temp(:)),zeros(G*G,1),c3*kappa2(kk,1)*eye(G*G))*p_j_SI(kk);
            u_ij2 = mvnpdf(ETA_vec(index_temp(:)),zeros(G*G,1),kappa2(kk,1)*eye(G*G))*(1-p_j_SI(kk));
            ost = u_ij2./(u_ij1 + u_ij2);
            gamma_SI(kk) = bernoullirnd(ost);
            GAMMA_SI(index_temp(:)) = gamma_SI(kk);
            VAR_SI(index_temp(:)) = kappa2(kk,1)*(c3^(1-gamma_SI(kk,1)));
        end
    end
    
    GAMMA_TEMP = reshape(GAMMA_SI,NG,NG);
    var_param = reshape(VAR_SI,NG,NG);
    for kk = 1:(NG-1)   
        omega{kk} = GAMMA_TEMP(kk+1,1:kk);
    end

    % Create PSI matrix from individual elements of "psi_ii_sq" and "eta"
    PSI_ALL = zeros(NG,NG);
    for nn_1 = 1:NG
        PSI_ALL(nn_1,nn_1) = sqrt(psi_ii_sq(nn_1,1));
    end

    for nn_2 = 1:NG-1
        eta_gg = cell2mat(eta(nn_2));
        for nnn = 1:size(eta_gg,1)
            PSI_ALL(nnn,nn_2+1) = eta_gg(nnn);
        end
    end
    
    % ========| Save post-burn-in draws
    if irep > nburn
        alpha_draws(irep-nburn,:) = alpha;
        sigma_draws(irep-nburn,:,:) = inv(PSI_ALL*PSI_ALL');
        gammaDI_draws(irep-nburn,:) = gamma_DI;        
        gammaCS_draws(irep-nburn,:) = gamma_CS;
        gammaSI_draws(irep-nburn,:) = gamma_SI;
    end
end

% Do some thinning
alpha_draws = alpha_draws(1:10:end,:);
sigma_draws = sigma_draws(1:10:end,:,:);
gammaDI_draws = gammaDI_draws(1:10:end,:);
gammaCS_draws = gammaCS_draws(1:10:end,:);
gammaSI_draws = gammaSI_draws(1:10:end,:);


gamma_DI_MC = [gamma_DI_MC, mean(gammaDI_draws,1)'];
gamma_CS_MC = [gamma_CS_MC, mean(gammaCS_draws,1)'];
gamma_SI_MC = [gamma_SI_MC, mean(gammaSI_draws,1)'];

alpha_MCMC_MC = [alpha_MCMC_MC, mean(alpha_draws,1)'];
sigma_MCMC_MC = [sigma_MCMC_MC, vec(mean(sigma_draws,1))];

alpha_OLS_MC = [alpha_OLS_MC, alpha_OLS_vec];
sigma_OLS_MC = [sigma_OLS_MC, vec(sigma_OLS)];
end

toc;