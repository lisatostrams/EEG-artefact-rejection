function [A,Lam,Nit,Cls]=...
    acdc(M,w,A0,Lam0,Nc,Tol);

%acdc: appoximate joint diagonalization
%(in the direct Least-Squares sense) of 
%a set of Hermitian matrices, using the
%iterative AC-DC algorithm.
%
%the basic call:
%[A,Lam]=acdc(M);
%
%Inputs:
%
%M(N,N,K) - the input set of K NxN 
%           "target matrices". Note that
%           all matrices must be 
%           symmetric (but need not be 
%           positive-definite).
%
%Outputs:
%
%A(N,Nc)   - the diagonalizing matrix.
%            by default, it is a square matrix (Nc=N),
%            however, the algorithm can also be used 
%            with Nc<M, e.g. for the over-complete case.
%            such values of Nc can be imposed either explicitly
%            (as an additional input parameter) or implicitly,
%            by providing an initial guess (see below) A0
%            of dimensions N by Nc, or Lam0 of dimensions 
%            Nc by K.
%
%Lam(Nc,K) - the diagonal values of the K
%            diagonal matrices.
%
%The algorithm finds an NxNc matrix A and
%K diagonal matrices
%         L(:,:,k)=diag(Lam(:,k))
%such that
% C_{LS}=
% \sum_k\|M(:,:,k)-A*L(:,:,k)*A'\|_F^2
%is minimized.
%
%-----------------------------------------
%   Optional additional input/output
%   parameters:
%-----------------------------------------
%
%[A,Lam,Nit,Cls]=
%	acdc(M,w,A0,Lam0,Nc,Tol);
%
%(additional) Inputs:
%
%w(K) - a set of positive weights such that
%       C_{LS}=
%       \sum_k\w(k)|M(:,:,k)-A*L(:,:,k)*A'\|_F^2
%       default: w=ones(K,1);
%
%A0 - an initial guess for A
%     default: A0=eye(N) (if no Nc, or if Nc=N; otheriwse, just
%     the first Nc columns of eye(N));
%
%Lam0 - an initial guess for the values of
%       Lam. If specified, an AC phase is
%       run first; otherwise, a DC phase is
%       run first.
%
%Nc   - the desired number of columns in A. Nc can be
%       smaller than N, e.g., for over-complete cases (N
%       sensors, Nc sources).
%       default: Nc=N.
%
%Tol  - An imposed tolerance on the change in C_{LS} for
%       a stopping condition.
%       default: Tol=TOL (see below)
%
%(additional) Outputs:
%
%Nit - number of full iterations
%
%Cls - vector of Nit Cls values
%
%-----------------------------------------
% Additional fixed processing parameters
%-----------------------------------------
%
%TOLF - a tolerance value on the change of
%       C_{LS}. AC-DC stops when the
%       decrease of C_{LS} is below a tolerance
%       originally set to:
%             TOLF/(N*N*sum(w));
%       which is reasonable, if the elements
%       in the matrices are of order ~1.
%       if some of the (weighted) matrices are 
%       exceptionally-scaled (see EXSC), a warning is 
%       generated, prompting the user to input a 
%       user-selected Tol (see above).
%
%EXSC - definition of "exceptional scale": if the
%       mean absolute value of the elements of any target
%       matrix is larger than EXSC or smaller than 1/EXSC,
%       and no user-defined Tol has been requested, a 
%       warning is generated (see TOL above)
%
%MAXIT - maximum number of allowed full
%        iterations.
%        Originally set to: 50;
%
%INTLC - number of AC sweeps to interlace
%        dc sweeps.
%        Originally set to: 1.
%
%-----------------------------------------
%
%Note that the implementation here is
%somewhat wasteful (computationally),
%mainly in performing a full eigenvalue
%decomposition at each AC iteration, 
%where in fact only the largest eigenvalue
%(and associated eigenvector) are needed,
%and could be extracted e.g. using the 
%power method. However, for small N (<10),
%the matlab eig function runs faster than
%the power method, so we stick to it.

%-----------------------------------------
%version R1.0, June 2000.
%By Arie Yeredor  arie@eng.tau.ac.il
%
%rev. R1.1, December 2001
%forced s=real(diag(S)) rather than just s=diag(S)
%in the AC phase. S is always real anyway; however,
%it may be set to a complex number with a zero 
%imaginary part, in which case the following
%max operation yields the max abs value, rather
%than the true max. This fixes that problem. -AY
%
%rev R1.2, March 2004
%enabled an over-complete version and a user-defined
%stopping threshold.
%
%Permission is granted to use and 
%distribute this code unaltered. You may 
%also alter it for your own needs, but you
%may not distribute the altered code 
%without obtaining the author's explicit
%consent.
%comments, bug reports, questions 
%and suggestions are welcome.
%
%References:
%[1] Yeredor, A., Approximate Joint 
%Diagonalization Using Non-Orthogonal
%Matrices, Proceedings of ICA2000, 
%pp.33-38, Helsinki, June 2000.
%[2] Yeredor, A., Non-Orthogonal Joint 
%Diagonalization in the Least-Squares 
%Sense with Application in Blind Source
%Separation, IEEE Trans. On Signal Processing,
%vol. 50 no. 7 pp. 1545-1553, July 2002.


%-----------------------------------------
%  here's where the fixed processing-
%  parameters are set (and may be 
%  modified):
%-----------------------------------------
TOLF=1e-3;
EXSC=10;
MAXIT=500;
INTLC=1;

%-----------------------------------------
%   here's where the inputs are collected
%   and validated, and defaults are set.
%-----------------------------------------
[N N1 K]=size(M);
if N~=N1
    error('input matrices must be square');
end
if K<2
    error('at least two input matrices are required');
end

if exist('w','var') & ~isempty(w)
    w=w(:);
    if length(w)~=K
        error('length of w must equal K')
    end   
    if any(w<=0)
        error('all weights must be positive');
    end
else
    w=ones(K,1);
end

if exist('Nc','var') & ~isempty(Nc)
    Nc=round(Nc);
    if Nc<1 or Nc>N
        error('Nc must satisfy 1<=Nc<=N')
    end
    Nc_flag=1;
else
    Nc_flag=0;
    Nc=N;
end

if exist('A0','var') & ~isempty(A0)
    [N_A0,Nc_A0]=size(A0);
    if N_A0~=N
        error('A0 must have the same number of rows as the target matrices')
    end
    if Nc_flag
        if Nc_A0~=Nc
            error('A0 must have Nc columns')
        end
    else
        Nc=Nc_A0;
    end
    A0_flag=1;
else
    A0=eye(N);
    A0=A0(:,1:Nc);
    A0_flag=0;
end

if exist('Lam0','var') & ~isempty(Lam0)
    if A0_flag
        error('Can''t initialize both A and Lam')
    end
    [Nc_L0,K_L0]=size(Lam0);
    if Nc_flag
        if Nc_L0~=Nc
            error('each vector in Lam0 must have Nc elements')
        end
    else
        Nc=Nc_L0;
        A0=A0(:,1:Nc);
    end
    if K_L0~=K
        error('Lam0 must have K vectors')
    end
    skipAC=0;
else
    Lam0=zeros(Nc,K);
    skipAC=1;
end

if exist('Tol','var') & ~isempty(Tol)
else
    %test if any of the matrices is exceptionally
    %scaled
    for k=1:K
        Melk=M(:,:,k);
        mabs=mean(abs(Melk(:)));
        if mabs>EXSC
            warning('Exceptionally large-valued matrix encountered - the default TOL may be inappropriate (consider using Tol)');
        end
        if mabs<1/EXSC
            warning('Exceptionally small-valued matrix encountered - the default TOL may be inappropriate (consider using Tol)');
        end
    end
    Tol=TOLF/(N*N*sum(w));
end

%-----------------------------------------
%  and this is where we start working
%-----------------------------------------

Cls=zeros(MAXIT,1);
Lam=Lam0;
A=A0;
for Nit=1:MAXIT
    
    if ~skipAC
        
        %AC phase   
        for nsw=1:INTLC
            for l=1:Nc
                P=zeros(N);
                for k=1:K
                    D=M(:,:,k);
                    for nc=[1:l-1 l+1:Nc]
                        a=A(:,nc);
                        D=D-Lam(nc,k)*a*a';
                    end
                    P=P+w(k)*Lam(l,k)*D;
                end
                [V S]=eig(P);
                s=real(diag(S));     %R1.1 - ay
                [vix,mix]=max(s);
                if vix>0
                    al=V(:,mix);
                    %this makes sure the 1st nonzero
                    %element is positive, to avoid
                    %hopping between sign changes:
                    fnz=find(al~=0);
                    al=al*sign(al(fnz(1)));
                    lam=Lam(l,:);
                    f=vix/((lam.*lam)*w);
                    a=al*sqrt(f);
                else
                    a=zeros(N,1);
                end	
                A(:,l)=a;
            end	%sweep
        end		%interlaces
    end			%skip AC
    skipAC=0;
    
    %DC phase
    AtA=A'*A;
    AtA2=AtA.*conj(AtA);
    G=inv(AtA2);
    for k=1:K
        Lam(:,k)=G*diag(A'*M(:,:,k)*A);
        L=diag(Lam(:,k));
        D=M(:,:,k)-A*L*A';
        Cls(Nit)=Cls(Nit)+w(k)*sum(sum(D.*conj(D)));
    end
    
    if Nit>1
        if abs(Cls(Nit)-Cls(Nit-1))<Tol
            break
        end
    end
    
end
Cls=Cls(1:Nit);
