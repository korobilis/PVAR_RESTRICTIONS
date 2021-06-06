function [what,alph,llik] = kalfilt(y1,Z,Ht,Qt,m,p,t)
% Run the Kalman filter and then return what -- mean of w 

%Kalman filter code
Kkeep = zeros(m*t,p);
Lkeep = zeros(m*t,m);
Fkeep = zeros(p*t,p);
a = zeros(m,t+1);
v = zeros(p,t);
Pt = zeros(m,m);   
htemp = Ht;
llik = 0;
for i = 1:t
    ztemp = Z((i-1)*p+1:i*p,:);
    v(:,i) = y1(:,i) - ztemp*a(:,i);
    Ft = ztemp*Pt*ztemp' + htemp;
    Ftinv = inv(Ft);
    llik = llik + log(det(Ft)) + v(:,i)'*Ftinv*v(:,i);
    Fkeep ((i-1)*p+1:i*p,:) = Ftinv;
    Kt = Pt*ztemp'*Ftinv ;
    Kkeep((i-1)*m+1:i*m,:) = Kt;
    Ltt = eye(m) - Kt*ztemp;
    Lkeep((i-1)*m+1:i*m,:) = Ltt;
    a(:,i+1) = a(:,i) + Kt*v(:,i);
    Pt = Pt*Ltt' + Qt;
end
llik = -.5*llik;
%Backward recursion to evaluate rt and, thus, whatt*/
rt = zeros(m,t+1);
pm = p+m;
what = zeros(pm*t,1);
htemp = Ht;
for i = t:-1:1
    ztemp = Z((i-1)*p+1:i*p,:);
    lterm = Lkeep((i-1)*m+1:i*m,:);
    fterm = Fkeep((i-1)*p+1:i*p,:)';
    kterm = Kkeep((i-1)*m+1:i*m,:);
    what((i-1)*pm+1:(i-1)*pm+p,1) = htemp*fterm*v(:,i) - htemp*kterm'*rt(:,i+1);
    what((i-1)*pm+p+1:i*pm,1) = Qt*rt(:,i+1);
    rt(:,i) = ztemp'*fterm*v(:,i) + lterm'*rt(:,i+1);
end

alph = zeros(m,t+1);
for i = 1:t
    alph(:,i+1) = alph(:,i) + Qt*rt(:,i+1);
end