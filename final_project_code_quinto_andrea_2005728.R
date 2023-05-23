# for some reason that I can't explain, setting a seed before splitting the data and befor running the
# bootstrap, arises an error even if the single code inside the cicle works if run once at the time
# for this reason the seeds are commented and the result can slightly differ from run to run

rm(list = ls())
library(ggplot2)
# Loading the data

load("data_final_project.RData")

num_list <- c('age', 'education', 'depression_scale', 'mod_vig_pa', 'sr_poor_health')
for (var in num_list) {
  idx_num <- which(colnames(data) == var)
  data[,idx_num] <- as.integer(data[,idx_num])
}
data$ferritin <- as.numeric(data$ferritin)

fac <- sapply(data, is.factor)
int <- sapply(data, is.integer)
num <- xor(sapply(data, is.numeric), int)

n_obs             <- nrow(data)
n_vars            <- ncol(data)
classes           <- data.frame(cbind(colnames(data), sapply(data, class)), row.names = NULL)
colnames(classes) <- c('FEATURE','CLASS')
fac               <- sapply(data, is.factor)
int               <- sapply(data, is.integer)
num               <- xor(sapply(data, is.numeric), int)

# Function fot the splitting of the original dataset in test set and training set with stratification of the outcome: # nolint

stratify <- function(dataframe, frac, replacement) {
  idx_zero       <- which(dataframe$outcome == 0)
  idx_one        <- which(dataframe$outcome == 1)
  idx_zero_train <- sample(idx_zero, round(frac * length(idx_zero)), replace = replacement)
  idx_zero_test  <- setdiff(idx_zero, idx_zero_train)
  idx_one_train  <- sample(idx_one, round(frac * length(idx_one)), replace = replacement)
  idx_one_test   <- setdiff(idx_one, idx_one_train)
  train_set      <- dataframe[c(idx_zero_train, idx_one_train),]
  test_set       <- dataframe[c(idx_zero_test, idx_one_test),]
  return(list(train_set, test_set))
}
#set.seed(123)
sets      <- stratify(data, frac = 0.8, replacement = FALSE)
train_set <- sets[[1]]
test_set  <- sets[[2]]

na_count              <- data.frame(colnames(data), row.names = NULL)
colnames(na_count)[1] <- 'FEATURES'
na_count[,2]          <- 100*apply(is.na(data),2 , sum)/dim(data)[1]
colnames(na_count)[2] <- 'FULL'
na_count[,3]          <- 100*apply(is.na(train_set),2 , sum)/dim(train_set)[1]
colnames(na_count)[3] <- 'TRAIN'


# Checking for colinearity

library(corrplot)
corr_mat <- cor(na.omit(train_set[,!fac]), na.omit(train_set[,!fac]))

corrplot(corr_mat, type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

ggplot(train_set, aes(ldl, tot_chol))+geom_point(colour = 'purple')+geom_smooth(method = lm, se = F, colour = 'black')
ggplot(data, aes(waist, bmi))+geom_point(colour = 'purple')+geom_smooth(method = lm, se = F, colour = 'black')
# being higly correlated, could be good to remove one of the two

idx_ldl <- which(colnames(train_set) == 'ldl')
idx_waist <- which(colnames(train_set) == 'waist')
train_set <- train_set[,-c(idx_ldl,idx_waist)]
test_set <- test_set[,-c(idx_ldl,idx_waist)]

fac <- fac[-c(idx_ldl, idx_waist)]
int <- int[-c(idx_ldl, idx_waist)]
num <- num[-c(idx_ldl, idx_waist)]
na_count <- na_count[-c(idx_ldl, idx_waist),]
n_vars <- ncol(train_set)


# Checking if there are extreme data points

# replacing the outliers with NAs
train_set_clean <- train_set
crprot_out <- which(train_set$crprot>100)
trig_out <- which(train_set$trig>1200)
ferritin_out <- which(train_set$ferritin>1200)
train_set_clean$crprot[crprot_out] <- NA
train_set_clean$trig[trig_out] <- NA
train_set_clean$ferritin[ferritin_out] <- NA
var_out <- c('crprot','trig','ferritin')

na_count[,4] <- 100*apply(is.na(train_set_clean),2 , sum)/dim(train_set_clean)[1]
colnames(na_count)[4] <- 'CLEANED'
na_count[,5]          <- 100*apply(is.na(test_set),2 , sum)/dim(test_set)[1]
colnames(na_count)[5] <- 'TEST'

variable <- rep(colnames(train_set),4)
dataset <- c(rep('full',23),rep('train',23),rep('cleaned',23),rep('test',23))
perc <- c(na_count$FULL,na_count$TRAIN,na_count$CLEANED,na_count$TEST)
nas <- data.frame(variable, dataset, perc)
ggplot(nas, aes(fill=dataset, y=perc, x=variable)) +
  geom_bar(position="dodge", stat="identity") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# just plotting the differneces
plot(train_set$crprot, main = 'crprot', col = 'green', pch = 19)
points(crprot_out, train_set$crprot[crprot_out], col = 'red', pch = 19)
plot(train_set$trig, main = 'trig', col = 'green', pch = 19)
points(trig_out, train_set$trig[trig_out], col = 'red', pch = 19)
plot(train_set$ferritin, main = 'ferritin', col = 'green', pch = 19)
points(ferritin_out, train_set$ferritin[ferritin_out], col = 'red', pch = 19)


# Data imputation

impute <- function(train, test, num) {
  for (i in 1:dim(train)[2]) {
    if (num[i]) {
      train[is.na(train[,i]),i] <- mean(na.omit(train[,i]))
      test[is.na(test[,i]),i] <- mean(na.omit(train[,i]))
    }
    if (!num[i]) {
      tab <- as.vector(table(train[,i]))
      all_vals <- sort(unique(train[,i]))
      idx_max <- which(tab==max(tab))
      train[is.na(train[,i]),i] <- all_vals[idx_max]
      test[is.na(test[,i]),i] <- all_vals[idx_max]
    }
  }
  return(list(train, test))
}
imp <- impute(train_set_clean, test_set, num)
train_imp <- imp[[1]]
test_imp <- imp[[2]]

na_count[,6] <- apply(is.na(train_imp),2 , sum)
colnames(na_count)[6] <- 'TRAIN IMP'

na_count[,7] <- apply(is.na(test_imp),2 , sum)
colnames(na_count)[7] <- 'TEST IMP'


# normalizing values of numerical values between 0 and 1

normalize <- function(train, test, fac, n_vars) {
  train_norm <- train
  test_norm <- test
  for (i in 1:n_vars) {
    if (!fac[i]) {
      train_norm[,i] <- (train[,i]-min(train[,i]))/(max(train[,i])-min(train[,i]))
      test_norm[,i] <- (test[,i]-min(train[,i]))/(max(train[,i])-min(train[,i]))
    }
  }
  return(list(train_norm, test_norm))
}

norm <- normalize(train_imp, test_imp, fac, n_vars)
train_norm <- norm[[1]]
test_norm <- norm[[2]]

train_matrix <- data.matrix(train_norm, rownames.force = NA)
summary(train_matrix[,fac])

model_full <- glm(outcome ~ ., data = train_norm, family = "binomial")
summary(model_full)

library(pROC)
pred_full <- predict(model_full, test_norm, type='response')
roc_full  <- roc(response = test_norm$outcome, predictor = pred_full, auc = TRUE)
plot.roc(roc_full, print.auc = TRUE, print.thres = TRUE, col = 'purple')


# BOOTSTRAP feature selection

n_boot <- 100
auc_full_vec <- 1:n_boot
auc_rfe_vec <- 1:n_boot
sel_vars <- list()
#set.seed(123)
# testing the average performance of the model
for (j in 1:n_boot) {
  # split into train and test
  sets      <- stratify(train_set_clean, frac = 1, replacement = TRUE)
  train_int <- sets[[1]]
  test_int  <- sets[[2]]
  # impute NAs
  imp <- impute(train_int, test_int, num)
  train_int_imp <- imp[[1]]
  test_int_imp <- imp[[2]]
  # normalize features
  norm_int <- normalize(train_int_imp, test_int_imp, fac, n_vars)
  test_int_norm  <- norm_int[[2]]
  train_int_norm <- norm_int[[1]]
  # recursive features elimination
  model_full_int  <- glm(outcome ~ ., data = train_int_norm, family = "binomial")
  pred_full_int   <- predict(model_full_int, test_int_norm, type='response')
  roc_full_int    <- roc(response = test_int_norm$outcome, predictor = pred_full_int, auc = TRUE, quiet = TRUE)
  auc_full_vec[j] <- roc_full_int$auc

  model_null_int  <- glm(outcome ~ 1, data = train_int_norm, family = "binomial")
  model_range_int <- list(lower = formula(model_null_int), upper = formula(model_full_int))
  rfe_int         <- step(object = model_full_int, scope = model_range_int, direction = "both")
  #print(summary(rfe_int))
  rfe_pred       <- predict(rfe_int, newdata = test_int_norm, type = "response")
  rfe_roc        <- roc(response = test_int_norm$outcome, predictor = rfe_pred, auc = TRUE)
  auc_rfe_vec[j] <- rfe_roc$auc

  sel_vars <- c(sel_vars,colnames(rfe_int$model))

}

print(mean(auc_full_vec))
print(mean(auc_rfe_vec))

counts <- data.frame(row.names = colnames(train_set[,-which(colnames(train_set) == 'outcome')]))
for (i in 1:dim(train_set)[2]) {
  if (!colnames(train_set)[i] == 'outcome') {
    counts[i,1] <- 100*sum(sel_vars == rownames(counts)[i]) / n_boot
    print(i)
    print(rownames(counts)[i])
  }
}

colnames(counts) <- '%'
print(counts)
write.csv(counts,"counts_rfe.csv", row.names = TRUE)

idx_sel <- which(counts[,1]>=75)

final_formula <- as.formula(paste("outcome", paste(colnames(train_set)[idx_sel], collapse = "+"), sep = "~"))
final_red_model <- glm(final_formula, data = train_norm, family = "binomial")
summary(final_red_model)
final_red_pred <- predict(final_red_model, test_norm, type = "response")
final_roc <- roc(response = test_norm$outcome, predictor = final_red_pred, auc = TRUE)
plot.roc(final_roc, print.auc = TRUE, print.thres = TRUE, col = 'purple')

# lasso
lambda_vec <- unique(c(seq(from = 0.0001, to = 0.001, by = 0.0001),
                       seq(from = 0.001, to = 0.01, by = 0.001),
                       seq(from = 0.01, to = 0.1, by = 0.01)))

library(caret)
library(glmnet)
auc_df <- data.frame(row.names = lambda_vec)
K <- 5

idx_0_train  <- which(train_set_clean$outcome == 0)
idx_0_fold_1 <- sample(idx_0_train, round(length(idx_0_train)/K), replace = FALSE)
idx_0_fold_2 <- sample(setdiff(idx_0_train, idx_0_fold_1), round(length(idx_0_train)/K), replace = FALSE)
idx_0_fold_3 <- sample(setdiff(idx_0_train, c(idx_0_fold_1,idx_0_fold_2)), round(length(idx_0_train)/K), replace = FALSE)
idx_0_fold_4 <- sample(setdiff(idx_0_train,c(idx_0_fold_1,idx_0_fold_2,idx_0_fold_3)), round(length(idx_0_train)/K), replace = FALSE)
idx_0_fold_5 <- setdiff(idx_0_train,c(idx_0_fold_1,idx_0_fold_2,idx_0_fold_3,idx_0_fold_4))
idx_0_fold   <- list(idx_0_fold_1,idx_0_fold_2,idx_0_fold_3,idx_0_fold_4,idx_0_fold_5)
for (i in 1:length(idx_0_fold)) {print(length(idx_0_fold[[i]]))}

idx_1_train  <- which(train_set_clean$outcome == 1)
idx_1_fold_1 <- sample(idx_1_train, round(length(idx_1_train)/K), replace = FALSE)
idx_1_fold_2 <- sample(setdiff(idx_1_train, idx_1_fold_1), round(length(idx_1_train)/K), replace = FALSE)
idx_1_fold_3 <- sample(setdiff(idx_1_train, c(idx_1_fold_1,idx_1_fold_2)), round(length(idx_1_train)/K), replace = FALSE)
idx_1_fold_4 <- sample(setdiff(idx_1_train, c(idx_1_fold_1,idx_1_fold_2,idx_1_fold_3)), round(length(idx_1_train)/K), replace = FALSE)
idx_1_fold_5 <- setdiff(idx_1_train, c(idx_1_fold_1, idx_1_fold_2, idx_1_fold_3, idx_1_fold_4))
idx_1_fold   <- list(idx_1_fold_1,idx_1_fold_2,idx_1_fold_3,idx_1_fold_4,idx_1_fold_5)
for (i in 1:length(idx_1_fold)) {print(length(idx_1_fold[[i]]))}

cocco <- 1
idx_outcome <- which(colnames(train_set) == "outcome")

for (lambda in lambda_vec) {

  for (k in 1:K) {

    idx_0_act_fold <- unlist(idx_0_fold[k])
    idx_1_act_fold <- unlist(idx_1_fold[k])
    train_int_clean <- train_set_clean[-c(idx_0_act_fold,idx_1_act_fold),]
    data_test_int  <- train_set_clean[c(idx_0_act_fold,idx_1_act_fold),]

    # remove NAs
    imp <- impute(train_int_clean, data_test_int, num)
    train_int_imp <- imp[[1]]
    test_int_imp <- imp[[2]]

    # normalize features
    norm_int <- normalize(train_int_imp, test_int_imp, fac, n_vars)
    train_imp_norm_int  <- norm_int[[1]]
    test_imp_norm_int <- norm_int[[2]]

    lasso_model_int <- glmnet(x = data.matrix(train_imp_norm_int[,-idx_outcome]),
                              y = data.matrix(train_imp_norm_int$outcome),
                              family = "binomial", alpha = 1, lambda = lambda, standardize = FALSE)
    lasso_pred_int  <- predict(lasso_model_int, newx = data.matrix(test_imp_norm_int[,-idx_outcome]), type = "response")
    lasso_roc_int   <- roc(test_imp_norm_int$outcome, as.vector(lasso_pred_int), quiet = TRUE, auc = TRUE)
    auc_df[cocco,k] <- lasso_roc_int$auc

    print(paste(cocco,k,sep='.'))
    print(lasso_roc_int$auc)

  }

  cocco <- cocco+1

}

print(auc_df)

avg_auc_df <- data.frame(apply(auc_df,1,mean))
avg_auc_df[,2] <- lambda_vec
rownames(avg_auc_df) <- NULL
colnames(avg_auc_df) <- c('mean_AUC','lambda')
print(avg_auc_df)
plot(x = lambda_vec, y = avg_auc_df[,1], log = c('x','y'))
ggplot(avg_auc_df, aes(lambda, mean_AUC))+geom_point()+scale_x_continuous(trans='log10')


idx_max_auc <- which.max(avg_auc_df[,1])
idx_ok <- which(avg_auc_df[,1] > avg_auc_df[idx_max_auc, 1] - 0.005)
idx_final_lambda <- which.max(as.vector(rownames(avg_auc_df)[idx_ok]))
final_lambda <- as.numeric(avg_auc_df$lambda[idx_final_lambda])

plot(x = lambda_vec, y = avg_auc_df[,1], log = 'x', pch = 19, col = 'green')
plot(x = lambda_vec[idx_ok], y = avg_auc_df[idx_ok,1], log = 'x', pch = 19, col = 'green',
     xlab = 'lambda', ylab = 'mean AUC')
points(x = lambda_vec[idx_final_lambda], y = avg_auc_df[idx_final_lambda,1], log = 'x',
       pch = 19, col = 'purple')
points(x = lambda_vec[idx_max_auc], y = avg_auc_df[idx_max_auc,1], log = 'x',
       pch = 19, col = 'red')
legend(1e-4, 0.9080, legend = c('lambda','lambda max','lambda sel'),
       col = c('green','red','purple'), pch = 19)

final_lasso_model <- glmnet(x = data.matrix(train_norm[,-idx_outcome]),
                            y = data.matrix(train_norm$outcome),
                            family = "binomial", alpha = 1, lambda = final_lambda, standardize = FALSE)
plot(final_lasso_model$beta, col = "purple", pch = 19)
lasso_pred <- predict(final_lasso_model, newx = data.matrix(test_norm[,-idx_outcome]), type = "response")
lasso_roc <- roc(response = test_norm$outcome, predictor = as.vector(lasso_pred), quiet = TRUE, auc = TRUE)
plot.roc(lasso_roc, print.auc = TRUE, print.thres = TRUE, col = "purple")

print(final_lasso_model$beta)
colnames(data)[as.vector(final_lasso_model$beta) > 0]


## bootstrap lasso

lasso_counts <- data.frame(row.names = colnames(train_set[,-idx_outcome]))
set.seed(123)

for (j in 1:n_boot) {

  sets      <- stratify(train_set_clean, frac = 1, replacement = TRUE)
  train_int <- sets[[1]]
  test_int  <- sets[[2]]
  # impute NAs
  imp <- impute(train_int, test_int, num)
  train_int_imp <- imp[[1]]
  test_int_imp <- imp[[2]]
  #train_int_imp <- kNN(data = train_int, k = 10)
  #train_int_imp <- train_int_imp[1:n_vars]
  #test_int_imp  <- kNN(data = test_int, k = 10)
  #test_int_imp  <- test_int_imp[1:n_vars]
  # normalize features
  norm_int <- normalize(train_int_imp, test_int_imp, fac, n_vars)
  train_int_norm  <- norm_int[[1]]
  test_int_norm <- norm_int[[2]]

  lasso_model_int <- glmnet(x = data.matrix(train_int_norm[,-idx_outcome]),
                            y = data.matrix(train_int_norm$outcome),
                            family = "binomial", alpha = 1, lambda = final_lambda, standardize = FALSE)

  lasso_counts[,j] <- as.vector(lasso_model_int$beta > 0)

}

lasso_counts <- data.frame(apply(lasso_counts, 1, sum))
print(lasso_counts)
colnames(lasso_counts) <- '%'
write.csv(lasso_counts,"counts_LASSO.csv", row.names = TRUE)
lasso_selected <- data.frame(rownames(lasso_counts)[lasso_counts[,1]>=75])
colnames(lasso_selected) <- 'FEAT'

formula_lasso <- as.formula(paste("outcome", paste(lasso_selected[,1], collapse = "+"), sep = "~"))
final_lasso <- glm(formula_lasso, data = train_norm, family = "binomial")
summary(final_lasso)
final_lasso_pred <- predict(final_lasso, test_norm, type = "response")
final_lasso_roc <- roc(response = test_norm$outcome, predictor = final_lasso_pred, auc = TRUE)
plot.roc(final_lasso_roc, print.auc = TRUE, print.thres = TRUE, col = 'purple')

## cv.glmnet
len_zero <- sum(train_norm[,idx_outcome] == 0)
len_one  <- sum(train_norm[,idx_outcome] == 1)
lasso_fit <- cv.glmnet(data.matrix(train_norm[,-idx_outcome]), data.matrix(train_norm[,idx_outcome]),
                       family = 'binomial', standardize = FALSE, keep = TRUE, nfolds = 10,
                       weights = c(rep(len_zero/len_one,len_one),rep(1,len_zero)))
plot(lasso_fit)
summary(lasso_fit)