library(parallel)

# load data saved in python with specific random_state

y_train <- read.csv("data/diabetes/y_train_diabetes.csv", header=F)
y_test <- read.csv("data/diabetes/y_test_diabetes.csv", header=F)
x_train <- read.csv("data/diabetes/x_train_diabetes.csv", header=F)
x_test <- read.csv("data/diabetes/x_test_diabetes.csv", header=F)

# set random seed
set.seed(42)

# define measures

res = data.frame(LL = NA, MSE = NA, time = NA)

nrsims <- 20
max_epochs <- 2000

Vs <- paste0("V",1:10)
form_mu <- paste0("~ 1 + s(V3)", 
                  # "+",
                  # paste(Vs, collapse=" + "),
                  #" + s(",
                  #paste(Vs[c(-2)], collapse=") + s("), ")", 
                  "+ ",
                  " dmu(",
                  paste(Vs, collapse=", "), ")")

form_sig <- paste0("~ 1 + ", "dsig(",
                   paste(Vs, collapse=", "), ")")

deep_mod <- function(x) x %>% 
  #layer_dense(units = 16, activation = "tanh", use_bias = FALSE) %>%
  layer_dense(units = 4, activation = "tanh") %>% 
  layer_dense(units = 1, activation = "linear")

### SSDR (w/ OZ)
res1 <- mclapply(1:nrsims, function(sim_iteration){
  
  # load deepregression
  library(deepregression)
  
  mod_deep <- deepregression(y = y_train$V1, 
                             list_of_formulas = list(loc = as.formula(form_mu),
                                                     scale = as.formula(form_sig)),
                             list_of_deep_models = list(dmu = deep_mod, 
                                                        dsig = deep_mod),
                             data = x_train,
                             family = "normal",
                             orthog_options = orthog_control(orthogonalize = TRUE),
                             tf_seed = sim_iteration
                             )
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = max_epochs)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(x_test)
  this_dist <- mod_deep %>% get_distribution(x_test, force_float = T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(y_test$V1, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  
  
  (mse <- (mean((pred-y_test$V1)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}, mc.cores = nrsims)

# get performance and times
res1 <- apply(do.call("rbind",res1), 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))
# LL        MSE      time
# [1,] 5.332936847 2484.84365 11.730768
# [2,] 0.004359729   22.20506  1.029578

res = data.frame(LL = NA, MSE = NA, time = NA)

### SSDR (w/o OZ)
res2 <- mclapply(1:nrsims, function(sim_iteration){
  
  # load deepregression
  library(deepregression)
  
  mod_deep <- deepregression(y = y_train$V1, 
                             list_of_formulas = list(loc = as.formula(form_mu),
                                                     scale = as.formula(form_sig)),
                             list_of_deep_models = list(dmu = deep_mod, 
                                                        dsig = deep_mod),
                             data = x_train,
                             family = "normal",
                             orthog_options = orthog_control(orthogonalize = FALSE),
                             tf_seed = sim_iteration
  )
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = max_epochs)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(x_test)
  this_dist <- mod_deep %>% get_distribution(x_test, force_float = T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(y_test$V1, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  
  
  (mse <- (mean((pred-y_test$V1)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}, mc.cores = nrsims)

# get performance and times
res2 <- apply(do.call("rbind",res2), 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))

form_mu <- paste0("~ 1", 
                  # "+",
                  #paste(Vs, collapse=" + "), 
                  #" + s(",
                  #paste(Vs[c(-2)], collapse=") + s("), ")", 
                  "+ ",
                  " dmu(",
                  paste(Vs, collapse=", "), ")")

### DNN
res3 <- mclapply(1:nrsims, function(sim_iteration){
  
  # load deepregression
  library(deepregression)
  
  mod_deep <- deepregression(y = y_train$V1, 
                             list_of_formulas = list(loc = as.formula(form_mu),
                                                     scale = ~1),
                             list_of_deep_models = list(dmu = deep_mod),
                             data = x_train,
                             family = "normal",
                             tf_seed = sim_iteration
  )
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = max_epochs)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(x_test)
  this_dist <- mod_deep %>% get_distribution(x_test, force_float = T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(y_test$V1, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  
  
  (mse <- (mean((pred-y_test$V1)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}, mc.cores = nrsims)

# get performance and times
res3 <- apply(do.call("rbind",res3), 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))
# LL        MSE      time
# [1,] 5.332936847 2484.84365 11.730768
# [2,] 0.004359729   22.20506  1.029578

write.csv(cbind(as.data.frame(rbind(res1,res2,res3)), method=rep(c("ssdrw","ssdrwo","dnn"), each=2)), 
                file="results_diabetes.csv")