function [MX,MODEL,F,Pl,BetP,L,EBIC] = EGMM(X, K, init,version)
%% Evidential Gaussian Mixture Model
% [MX MODEL] = EGMM(X, K, version)
%  Input:
%  - X: N-by-D data matrix
%  - K: the number of clusters
%  - init: 0:  initialize randomly, 1: initialize using k-means
%  - version : 0:  2^K-1 focal elements, 1: focal elements of
%       size less or equal to 2 except Omega 
%  Output:
%  - MX: N-by-M matrix indicating the evidential membership of each sample to each focal element
%  - MODEL: a structure containing the parameters for a EGMM:
%       MODEL.Miu: a M-by-D matrix
%       MODEL.Sigma: a D-by-D matrix
%       MODEL.Pi: a 1-by-M vector
%  - F: M-by-K matrix indicating the focal elements
%       F(i,j)=1 if omega_j belongs to focal element i
%              0 otherwise
%   - Pl: N-by-K matrix indicating the plausibilities of each sample to each cluster
%   - BetP: N-by-K matrix indicating the pignistic probabilities of each sample to each cluster
%   - L: value of evidential Gaussian mixture log-likelihood 
%   - EBIC:  value of evidential Bayesian inference criteria (cluster validity index)
% Copyright (C) 2020-2022 by Lianmeng JIAO
% Version 1.0 -- 2020/4/8
%% 
threshold = 1e-3;
Lprev = -inf;
[N, D] = size(X);

%--------------- construction of the focal set matrix F ---------
ii=1:2^K;
F=zeros(length(ii),K);
for i=1:K
      F(:,i)=bitget(ii'-1,i);
end
F(1,:)=[]; % without empty set

if version==0 
    M = 2^K - 1; % number of focal elements without empty set
end

if version==1  % limitation of focal sets to cardinality <=2 + Omega
    if K>3
        truc = sum(F,2);
        ind = find(truc>2);
        ind(end)=[];
        F(ind,:)=[];
    end

    if K~=2
        M = K + K*(K-1)/2 + 1 ; % without empty set
    else
        M = 3;
    end
end
c = sum(F,2)';  % cardinality of focal sets

%---------------------- initialisations--------------------------------------
% get the initialized prior prabability pPi, mean vectors pMiu of the K clusters and common covariance matrices pSigma 
if init==0
    [ pPi pMiu pSigma] = init_params_random();
else
    [ pPi pMiu pSigma] = init_params_kmeans();
end

% calculate the mean vectors pMiu_plus and covariance matrices pSigma_plus of the M evidential components
pMiu_plus=zeros(M,D);
pSigma_plus = zeros(D,D,M);
for j=1:M
    fj = F(j,:);
    truc = repmat(fj',1,D);
    pMiu_plus(j,:) = sum(pMiu.*truc)./sum(truc);
    pSigma_plus(:,:,j) = pSigma;
end

%------------------------ iterations--------------------------------
num = 0;
while true
    % calculate the evidential memberships using the current parameters
    Px = zeros(N, M);
    for j = 1:M
        Xshift = X-repmat(pMiu_plus(j, :), N, 1);
        inv_pSigma_plus = inv(pSigma_plus(:,:,j)+0.01*eye(D));
        tmp = sum((Xshift*inv_pSigma_plus).* Xshift, 2);
        coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma_plus));
        Px(:, j) = coef * exp(-0.5*tmp);
    end
    
    MX = Px .* repmat(pPi, N, 1);
    MX = MX ./ repmat(sum(MX, 2), 1, M);
    
    % Update parameters of each cluster (prior prabability pPi, mean pMiu and covariance pSigma)
    Nj = sum(MX, 1);
    pPi = Nj/N; % prior prabability pPi
    
    H = zeros(K,K);
    for k=1:K
        for l=1: K
            truc = zeros(1,K);
            truc(1,k)=1;truc(1,l)=1;
            t = repmat(truc,M,1) ;
            indices = find(sum((F-t)-abs(F-t),2)==0) ;   % indices of all Aj containing wk and wl
            for jj = 1:length(indices)
                j = indices(jj);
                H(l,k)=H(l,k)+sum(MX(:,j))*c(j)^(-2);
            end
        end
    end
    B=[];
    for l=1:K
        truc = zeros(1,K);
        truc(1,l)=1;
        t = repmat(truc,M,1) ; 
        indices = find(sum((F-t)-abs(F-t),2)==0) ;   % indices of all Aj containing wl
        mi = repmat((c(indices).^(-1)),N,1).*MX(:,indices);
        s = sum(mi,2);
        mats = repmat(s,1,D);
        xim = X.*mats ;
        blq = sum(xim);
        B=[B;blq];
    end
    pMiu=H\B; % mean pMiu
    
    pMiu_plus=zeros(M,D);
    for j=1:M
        fj = F(j,:);
        truc = repmat(fj',1,D);
        pMiu_plus(j,:) = sum(pMiu.*truc)./sum(truc);
    end
    pSigmaj = zeros(D,D,M);
    for j = 1:M
        Xshift = X-repmat(pMiu_plus(j, :), N, 1);
        pSigmaj(:, :, j) = (Xshift' * (diag(MX(:, j)) * Xshift)) / N;
    end
    pSigma = sum(pSigmaj,3); % covariance pSigma
    
    % calculate the mean vectors pMiu_plus and covariance matrices pSigma_plus of the M evidential components
    for j=1:M
        pSigma_plus(:,:,j) = pSigma;
    end
    
    % check for convergence
    Px = zeros(N, M);
    for j = 1:M
        Xshift = X-repmat(pMiu_plus(j, :), N, 1);
        inv_pSigma_plus = inv(pSigma_plus(:,:,j)+0.01*eye(D));
        tmp = sum((Xshift*inv_pSigma_plus).* Xshift, 2);
        coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma_plus));
        Px(:, j) = coef * exp(-0.5*tmp);
    end
    L = sum(log(Px*pPi'));
    
    if L-Lprev < threshold
        break;
    end
    Lprev = L;
    num = num + 1;
%     fprintf('iteration %d: log-likelihood = %6.3f\n',num, L);
%     pMiu_plus
end

%------------------------ prepare the output--------------------------------
MODEL = [];
MODEL.Miu = pMiu_plus;
MODEL.Sigma = pSigma;
MODEL.Pi = pPi;

MXp = [zeros(N,1), MX];  %with the empty set
ii=1:2^K;
FF=zeros(length(ii),K);
for i=1:K
    FF(:,i)=bitget(ii'-1,i);
end
if version==1
    if K>3
        mm=zeros(N,2^K);
        truc = sum(FF,2);
        ind = find(truc<=2);
        ind = [ind;2^K];
        for j=1:length(ind)
            mm(:,ind(j))=MXp(:,j);
        end
    else
        mm=MXp;
    end
    P=[];
    BetP=[];
    for i=1:N
        pp = mtopl(mm(i,:));
        P=[P;pp];
        bet = betp(mm(i,:));
        BetP = [BetP;bet];
    end
else
    P=[];
    BetP=[];
    for i=1:N
        pp = mtopl(MXp(i,:));
        P=[P;pp];
        bet = betp(MXp(i,:));
        BetP = [BetP;bet];
    end
end

truc = sum(FF,2);
singletons = find(truc==1);
Pl = P(:,singletons);

VC = M-1 + K*D + D*(D+1)/2;
EBIC = L - VC*log(N)/2;  % evidential Bayesian inference criteria (cluster validity index)

%------------------------ base functions--------------------------------
function [pPi pMiu pSigma] = init_params_random()
    % parameters initialization randomly
    Idx_rand = randperm(N);
    pMiu = X(Idx_rand(1:K),:);
    pPi = ones(1, M)/M;
    pSigmak = zeros(D, D, K);
    
    % hard assign x to each centroids
    distmat = repmat(sum(X.*X, 2), 1, K) + ...
        repmat(sum(pMiu.*pMiu, 2)', N, 1) - ...
        2*X*pMiu';
    [dummy labels] = min(distmat, [], 2);
    
    for k=1:K
        Xk = X(labels == k, :);
        pSigmak(:, :, k) = cov(Xk);
    end
    pSigma = sum(pSigmak,3)/K;
end

function [ pPi pMiu pSigma] = init_params_kmeans()
    % parameters initialization by k-means
    [IDX,pMiu] = kmeans(X,K);
    
    pPi = ones(1, M)/M;
    pSigmak = zeros(D, D, K);
    ind = find(c==1);
    for k=1:K
        Xk = X(IDX == k, :);
        pSigmak(:, :, k) = cov(Xk);
    end
    pSigma = sum(pSigmak,3)/K;
end

function [out] = betp(in) 
% computing BetP on ? from the m vector (in) 
% out = BetP vector: order a,b,c,... 
% beware: not optimalize, so can be slow for >10 atoms 
 
lm = length(in); 
natoms = round(log2(lm)); 		 
if 2^natoms == lm  
	if in(1) == 1 
		out = 	ones(1,natoms)./natoms; 
	else 
		betp = zeros(1,natoms); 
		for i = 2:lm 
			x = bitget(i-1,1:natoms); % x contains the 1 and 0 of i-1, for a,b,c... 
			betp = betp + in(i)/sum(x).*x; 
		end 
		out = betp./(1-in(1)); 
	end 
else 
	'ACCIDENT in betp: length of input vector not OK: should be a power of 2' 
end 
end

function [out] = mtopl(in) 
% computing FMT from m to pl 
% in = m vector 
% out = pl vector 
 
in = mtob(in); 
out = btopl(in); 
end

function [out] = mtob(in) 
% computing FMT from m to b.  
% in = m vector 
% out = b vector 

lm = length(in); 
natoms = round(log2(lm)); 		 
if 2^natoms == lm
    for step = 1:natoms
        i124 = 2^(step-1);
        i842 = 2^(natoms+1-step);
        i421 = 2^(natoms - step);
        in = reshape(in,i124,i842);
        in(:,(1:i421)*2) = in(:,(1:i421)*2) + in(:,(1:i421)*2-1);
    end
    out = reshape(in,1,lm);
else
    'ACCIDENT in mtob: length of input vector not OK: should be a power of 2'
end
end

function [out] = btopl(in) 
% compute pl from b 
% in = b 
% out = pl 
 
lm = length(in); 
natoms = round(log2(lm)); 		 
if 2^natoms == lm
    in = in(lm)-fliplr(in);
    in(1) = 0;
    out = in;
else
    'ACCIDENT in btopl: length of input vector not OK: should be a power of 2'
end
end

end