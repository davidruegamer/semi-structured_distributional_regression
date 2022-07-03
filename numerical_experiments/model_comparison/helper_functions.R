char_to_fun <- function(char) eval(parse(text = paste0(char, '()')))

otl <- function(S,L)
{
  qrL <- qr(L)
  Q <- qr.Q(qrL)
  X_XtXinv_Xt <- tcrossprod(Q)
  Sorth <- S - X_XtXinv_Xt%*%S
  return(Sorth)
}

center_nl <- function(fun){
  otl(fun, matrix(rep(1,length(fun)), ncol=1))
}
