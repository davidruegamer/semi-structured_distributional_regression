library(parallel)

ff <- read.csv("data/forestfires/forestfires.csv")
# transform outcome as described in 
# the repository (https://archive.ics.uci.edu/ml/datasets/forest+fires)
ff$area <- log(ff$area + 1)
ff$month <- as.factor(ff$month)
ff$day <- as.factor(ff$day)

set.seed(42)

index_train <- sample(1:nrow(ff), round(nrow(ff)*0.75))

train <- ff[index_train,]
test <- ff[setdiff(1:nrow(ff), index_train),]
y_train <- train$area
y_test <- test$area

# define measures

res = data.frame(LL = NA, MSE = NA, time = NA)

nrsims <- 20
max_epochs <- 2000

# train <- model.matrix(~ 0 + ., data = train)
# test <- model.matrix(~ 0 + ., data = test)

# write.csv(cbind(as.data.frame(train),area=y_train), "~/SDDL/SDDL/distcal/data/forestfires/train.csv")
# write.csv(cbind(as.data.frame(test),area=y_test), "~/SDDL/SDDL/distcal/data/forestfires/test.csv")

Vs <- setdiff(colnames(train),"area")

form_mu <- paste0("~ 1", 
                   " + ", #
                  paste(Vs, collapse=" + "),
                  #"+",
                  # "s(",
                  # paste(Vs[-1*c(1:20,28)], collapse=") + s("), ")", 
                  " + deep_mu(",
                  paste(Vs, collapse=", "), ")")

form_sig <- paste0("~ 1",
                   "+",
                   "month"
                   #paste(Vs, collapse=" + ") #, 
                   # "+",
                   #  "deep_sig(",
                   #  paste(Vs, collapse=", "), ")"
                   )

deep_mod <- function(x) x %>% 
  layer_dense(units = 4, activation = "tanh", use_bias = FALSE) %>%
  layer_dense(units = 4, activation = "tanh") %>%
  layer_dense(units = 1, activation = "linear")

deep_mod2 <- function(x) x %>% 
  #layer_dense(units = 2, activation = "tanh", use_bias = FALSE) %>%
  # layer_dense(units = 4, activation = "tanh") %>%
  layer_dense(units = 1, activation = "linear")

### SSDR
res1 <- mclapply(1:nrsims, function(sim_iteration){
  
  # load deepregression
  library(deepregression)
  
  mod_deep <- deepregression(y = y_train, 
                             list_of_formulas = list(loc = as.formula(form_mu),
                                                     scale = as.formula(form_sig)),
                             list_of_deep_models = list(deep_mu = deep_mod, 
                                                        deep_sig = deep_mod2),
                             data = as.data.frame(train),
                             family = "normal",
                             orthog_options = orthog_control(orthogonalize = TRUE),
                             tf_seed = sim_iteration
                             )
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = max_epochs)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, 
                   view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(as.data.frame(test))
  this_dist <- mod_deep %>% get_distribution(as.data.frame(test), force_float = T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(y_test, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  (mse <- (mean((pred-y_test)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}, mc.cores = nrsims)

# get performance and times
res1 <- apply(do.call("rbind",res1), 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))
# LL        MSE       time
# [1,] 1.753718152 1.95265084 12.4660117
# [2,] 0.006806554 0.01573052  0.5644636

res = data.frame(LL = NA, MSE = NA, time = NA)

### SSDR (w/o orthog)
res2 <- mclapply(1:nrsims, function(sim_iteration){
  
  # load deepregression
  library(deepregression)
  
  mod_deep <- deepregression(y = y_train, 
                             list_of_formulas = list(loc = as.formula(form_mu),
                                                     scale = as.formula(form_sig)),
                             list_of_deep_models = list(deep_mu = deep_mod, 
                                                        deep_sig = deep_mod2),
                             data = as.data.frame(train),
                             family = "normal",
                             orthog_options = orthog_control(orthogonalize = FALSE),
                             tf_seed = sim_iteration
  )
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = max_epochs)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, 
                   view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(as.data.frame(test))
  this_dist <- mod_deep %>% get_distribution(as.data.frame(test), force_float = T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(y_test, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  (mse <- (mean((pred-y_test)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}, mc.cores = nrsims)

# get performance and times
res2 <- apply(do.call("rbind",res2), 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))
# LL        MSE       time
# [1,] 1.753718152 1.95265084 12.4660117
# [2,] 0.006806554 0.01573052  0.5644636

### DNN only

form_mu <- paste0("~ 1",
                  " + deep_mu(",
                  paste(Vs, collapse=", "), ")")

res = data.frame(LL = NA, MSE = NA, time = NA)

res3 <- mclapply(1:nrsims, function(sim_iteration){
  
  # load deepregression
  library(deepregression)
  
  mod_deep <- deepregression(y = y_train, 
                             list_of_formulas = list(loc = as.formula(form_mu),
                                                     scale = ~1),
                             list_of_deep_models = list(deep_mu = deep_mod),
                             data = as.data.frame(train),
                             family = "normal",
                             tf_seed = sim_iteration
  )
  
  st <- Sys.time()
  
  cvres <- mod_deep %>% cv(epochs = max_epochs)
  
  (ep <- stop_iter_cv_result(cvres))
  
  mod_deep %>% fit(epochs = ep, 
                   verbose = FALSE, 
                   view_metrics = FALSE,
                   validation_split = NULL)
  
  et <- Sys.time()
  
  pred <- mod_deep %>% predict(as.data.frame(test))
  this_dist <- mod_deep %>% get_distribution(as.data.frame(test), force_float = T)
  
  log_score_fun <- function(y,m,s) dnorm(y,m,s,log=T)
  
  (ll <- -mean(
    do.call(log_score_fun, list(y_test, 
                                as.matrix(this_dist %>% tfd_mean()),
                                as.matrix(this_dist %>% tfd_stddev())))
  ))
  
  (mse <- (mean((pred-y_test)^2)))
  
  res[sim_iteration, ] <- c(ll, mse, as.numeric(difftime(et,st,units="mins")))
  
}, mc.cores = nrsims)

# get performance and times
res3 <- apply(do.call("rbind",res3), 2, function(x) c(mean(x, na.rm=T), sd(x, na.rm=T)))

write.csv(cbind(as.data.frame(rbind(res1,res2,res3)), method=rep(c("ssdrw","ssdrwo","dnn"), each=2)), 
          file="results_forestf.csv")