% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/deep-ensembles.R
\name{ensemble.deepregression}
\alias{ensemble.deepregression}
\title{Ensemblind deepregression models}
\usage{
\method{ensemble}{deepregression}(
  x,
  n_ensemble = 5,
  reinitialize = TRUE,
  mylapply = lapply,
  verbose = FALSE,
  patience = 20,
  plot = TRUE,
  print_members = TRUE,
  stop_if_nan = TRUE,
  save_weights = TRUE,
  callbacks = list(),
  save_fun = NULL,
  ...
)
}
\arguments{
\item{x}{object of class \code{"deepregression"} to ensemble}

\item{n_ensemble}{numeric; number of ensemble members to fit}

\item{reinitialize}{logical; if \code{TRUE} (default), model weights are
initialized randomly prior to fitting each member. Fixed weights are
not affected}

\item{mylapply}{lapply function to be used; defaults to \code{lapply}}

\item{verbose}{whether to print training in each fold}

\item{patience}{number of patience for early stopping}

\item{plot}{whether to plot the resulting losses in each fold}

\item{print_members}{logical; print results for each member}

\item{stop_if_nan}{logical; whether to stop CV if NaN values occur}

\item{save_weights}{whether to save final weights of each ensemble member;
defaults to \code{TRUE}}

\item{callbacks}{a list of callbacks used for fitting}

\item{save_fun}{function applied to the model in each fold to be stored in
the final result}

\item{...}{further arguments passed to \code{object$fit_fun}}
}
\value{
object of class \code{"drEnsemble"}, containing the original
    \code{"deepregression"} model together with a list of ensembling
    results (training history and, if \code{save_weights} is \code{TRUE},
    the trained weights of each ensemble member)
}
\description{
Ensemblind deepregression models
}
