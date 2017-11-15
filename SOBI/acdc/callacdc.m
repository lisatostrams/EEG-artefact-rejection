%callacdc (call acdc)- a small demo 
%showing the way to call acdc (the 
%Hermitian version) with and 
%without initialization.
%More simple initializations (such as
%exact joint diagonalization of two
%of the matrices) are also possible.

%since real-valued matrices are used, the're
%no real difference in here between the Hermitian
%and the symmetric cases. However, this routine
%still demonstrates the call to the Hermitian
%version (acdc.m), which is indeed preferable
%for the real-valued case. See callacdc_sym.m
%foo a demo with the symmetric version (with
%complex-valued matrices).

%generate the target matrices.
%for running with old data set newdata=0.
newdata=1;
if newdata
    N=5;
    K=10;
    TrueA=randn(N,N);
    %the 1st matrix is p.d. to allow whitening
    TrueL=[rand(N,1) randn(N,K-1)];
    M=zeros(N,N,K);
    for k=1:K
        M(:,:,k)=TrueA*diag(TrueL(:,k))*TrueA';
        Noise=.1*randn(N,N);
        %Noise has to be Hermitian. Also, make
        %it p.d. to keep the 1st matrix p.d. for
        %the initialization
        Noise=Noise*Noise';
        M(:,:,k)=M(:,:,k)+Noise;
    end
end

%call acdc without initialization
A1=acdc(M);
%show the resulting "demixing":
disp('demixing attained by ACDC w/o initialization:')
disp(A1\TrueA)

%initialize for acdc
A0=init4acdc(M);
%show the demixing attained by A0:
disp('demixing attained by whitening + orthogonal diagonalization:')
disp(A0\TrueA)

A2=acdc(M,[],A0);
%show the demixing attained with proper initialization:
disp('demixing attained by ACDC (with initialization):')
disp(A2\TrueA)

