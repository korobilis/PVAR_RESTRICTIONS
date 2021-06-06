function [bdraw,vv] = carter_kohn_KP(y,Z,R,Qt,m,p,t,s,B0,V0)
% Carter and Kohn (1994), On Gibbs sampling for state space models.

% Kalman Filter
vv = zeros((t-1)*m,m);
bp = B0;
Vp = V0;
bt = zeros(t,m);
Vt = zeros(m^2,t);
for i=1:t
    H = Z((i-1)*p+1:i*p,:);
    cfe = y(:,i) - H*bp;   % conditional forecast error
    Rx = Vp*H';
    KV = H*Rx + R;    % variance of the conditional forecast error
    KG = Rx/KV;
    btt = bp + KG*cfe;
    Vtt = Vp - KG*(H*Vp);
    if i < t
        bp = btt;
        if i>1 && s(i-1)~=s(i);
            %if there is switch between t and t+1:
            Vp = Vtt + Qt; %Predicted state covariance       
        else
            %no switch - do not update predicted covariance with sigma
            Vp = Vtt; %Predicted state covariance
        end
    end

    bt(i,:) = btt';
    Vt(:,i) = reshape(Vp,m^2,1);
end

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
bdraw = zeros(t,m);
bdraw(s(t),:) = mvnrnd(btt,Vtt,1);

if s(t,:)<t;       
    for i = s(t)+1:t
        bdraw(i,:) = bdraw(i-1,:) + (chol(Qt)'*randn(m,1))';
        vv((i-2)*m+1:(i-1)*m,:) = (bdraw(i,:)-bdraw(i-1,:))'*(bdraw(i,:)-bdraw(i-1,:));
    end
end

% Backward recurssions
for i = t:-1:2
    if s(i) ~= s(i-1)
        bf = bdraw(s(i),:)';
        btt = bt(i-1,:)';
        Vtt = reshape(Vt(:,i-1),m,m);
        f = Vtt + Qt;
        K = Vtt/f;
        cfe = bf - btt;
        bmean = btt + K*cfe;
        bvar = Vtt - K*Vtt;
        bdraw(s(i-1),:) = bmean' + randn(1,m)*chol(bvar);  %mvnrnd(bmean,bvar,1);
        vv((s(i-1)-1)*m+1:s(i-1)*m,:) = (bdraw(s(i),:)-bdraw(s(i-1),:))'*(bdraw(s(i),:)-bdraw(s(i-1),:));        
    end
end
bdraw = bdraw';