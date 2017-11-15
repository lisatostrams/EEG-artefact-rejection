readme.txt
As of now (2-mar-04) this directory
contains the following:

* The acdc algorithm for finding the 
  approximate general (non-orthogonal) 
  joint diagonalizer (in the direct Least
  Squares sense) of a set of Hermitian 
  matrices.
  [acdc.m]

* The acdc algorithm for finding the 
  same for a set of Symmetric (rather than
  Hermitean) matrices.
  [acdc_sym.m]
  Note that for real-valued matrices the
  Hermitian and Symmetric cases are similar;
  however, in such cases the Hermitian version
  [acdc.m], rather than the Symmetric version
  [acdc_sym] is preferable.

* A function that finds an initial guess
  for acdc by applying hard-whitening
  followed by Cardoso's orthogonal joint
  diagonalizer. Note that acdc may also
  be called without an initial guess,
  in which case the initial guess is set 
  by default to the identity matrix.
  The m-file includes the joint_diag
  function (by Cardoso) for performing
  the orthogonal part.
  [init4acdc.m]

* A small routine that demonstrates the 
  call (with and without initialization)
  to the Hermitian vesion after generating 
  a set of target-matrices.
  [callacdc.m]

* A small routine that demonstrates the 
  same with the Hermitian vesion.
  [callacdc_sym.m]

* The acdc and acdc_sym codes have been revised
 (relative to the older version) in two aspects:
  + The overcomplete case (A has more rows than
    columns) has been made explicitly available,
    by introducing a new (optional) input parameter,
    Nc (the number of columns in A);
  + A threshold parameter (Tol) was added as another
    (optional) input parameter, to serve as a user-
    defined stopping criterion. If Tol is not 
    specified, then an automatic threshold is used,
    but a warning message is generated if the scales
    of some matrices appear incompatible with
    this threshold.


Contibuted By:
Dr. Arie Yeredor,
School of Electrical Engineering,
Tel-Aviv University.
e-mail: arie@eng.tau.ac.il
web-site: www.eng.tau.ac.il\~arie

comments, bug reports, questions 
and suggestions are welcome.

References:

[1] Yeredor, A., Approximate Joint 
Diagonalization Using Non-Orthogonal
Matrices, Proceedings of ICA2000, 
pp.33-38, Helsinki, June 2000.

[2] Yeredor, A., Non-Orthogonal Joint 
Diagonalization in the Least-Squares 
Sense with Application in Blind Source
Separation, IEEE Trans. On Signal Processing,
vol. 50 no. 7 pp. 1545-1553, July 2002.
 
