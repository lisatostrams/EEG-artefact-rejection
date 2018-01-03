subroutine elementwise( a, b, c, M, N ) bind(c, name='elementwise')
  use iso_c_binding, only: c_float, c_int

  integer(c_int),intent(in) :: M, N
  real(c_float), intent(in) :: a(M, N), b(M, N)
  real(c_float), intent(out):: c(M, N)

  integer :: i,j

  forall (i=1:M,j=1:N)
    c(i,j) = a(i,j) * b(i,j)
  end forall

end subroutine