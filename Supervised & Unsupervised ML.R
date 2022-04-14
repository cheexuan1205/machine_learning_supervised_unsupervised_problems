# © 2022 YUI CHEE XUAN
#Unsupervised Learning
#R code to import and prepare the EWCS dataset
ewcs=read.table("EWCS_2016.csv",sep=",",header=TRUE)
ewcs[,][ewcs[, ,] == -999] <- NA
# Identify complete rows of 'ewcs' data frame
kk=complete.cases(ewcs)
# Remove the rows with missing values, 'NA'
ewcs=ewcs[kk,]

names(ewcs)

# Explore the mean of variables
apply(ewcs, 2, mean)

# Explore the variance of variables
apply(ewcs, 2, var)

# Principal Components Analysis==========================
pca.out <- prcomp(ewcs, scale = TRUE)

names(pca.out)
# Mean of variables prior to PCA implementation
pca.out$center 
# Standard deviation of variables prior to PCA implementation
pca.out$scale
# Loading vectors of principal components based on variables
pca.out$rotation
dim(pca.out$x)

# Visualistion
biplot(pca.out, scale = 0)

# Standard deviation of principal components
pca.out$sdev

# Variance explained by each principal component
pr.var <- pca.out$sdev^2
pr.var

# Proportion of Variance Explained (PVE) by each principal component
pve <- pr.var / sum(pr.var)
round(pve,4)

# Cumulative Proportion of Variance Explained (CPVE)
cpve <- cumsum(pve)
round(cpve,4)

# Scree Plot Visualisation
# Split the screen in several panels
par(mfrow = c(1,2))
plot(pve, xlab = "Principal Component", 
    ylab = "Proportion of Variance Explained (PVE)",
    ylim = c(0,1), type = 'b', main = 'Scree Plot')
plot(cpve, xlab = "Principal Component",
    ylab = "Cumulative Proportion of Variance Explained (CPVE)",
    ylim = c(0,1), type = 'b', main = 'Cumulative Scree Plot')
# To ensure not splitting the screen in several panels next time
par(mfrow = c(1,1))

#======================================================================================
# ___________________
#|Regression Problem |
# ___________________
# Import required library
library(caTools)
# R code to import and prepare the student performance dataset
math=read.table("student-mat.csv",sep=";",header=TRUE)
portu=read.table("student-por.csv",sep=";",header=TRUE)

head(math)
summary(math)

head(portu)
summary(portu)

# Splitting dataset into train & test for math dataset
set.seed(1)
split_m <- sample.split(math, SplitRatio=0.7)
train_m <- subset(math, split_m=="TRUE")  #schools[split, ] 
test_m <- subset(math, split_m=="FALSE")  #schools[!split, ]

# Splitting dataset into train & test for portuguese language dataset
set.seed(1)
split_p <- sample.split(portu, SplitRatio=0.7)
train_p <- subset(portu, split_p=="TRUE")  
test_p <- subset(portu, split_p=="FALSE")  

#observe missing value
library(Amelia)
missmap(math, main = "Missing values vs observed")
missmap(portu, main = "Missing values vs observed")
dev.off()

# double checking for missing value
sapply(math,function(x) sum(is.na(x)))
sapply(portu,function(x) sum(is.na(x)))

# Multiple linear regression for math==================================
lm_fit_m <- lm(G3 ~ . - G2 - G1, data=train_m)
lm_fit_m
summ_m <- summary(lm_fit_m)
summ_m
names(lm_fit_m)
coef(lm_fit_m) # model coefficients for model parameters
confint(lm_fit_m) #confidence interval 
predict(lm_fit_m , test_m, interval = "confidence") #confidence interval
predict(lm_fit_m , test_m, interval = "prediction") #prediction interval, wider than confidence interval
fitted(lm_fit_m) # predicted values
residuals(lm_fit_m) # residuals
anova(lm_fit_m) # anova table
vcov(lm_fit_m) # covariance matrix for model parameters

#visualise
plot(lm_fit_m)

# Multiple linear regression for portuguese language=========================
lm_fit_p <- lm(G3 ~ . - G2 - G1, data=train_p)
lm_fit_p
summ_p <-summary(lm_fit_p)
summ_p
names(lm_fit_p)
coef(lm_fit_p) # model coefficients for model parameters
confint(lm_fit_p) #confidence interval 
predict(lm_fit_p , test_m, interval = "confidence") #confidence interval
predict(lm_fit_p , test_m, interval = "prediction") #prediction interval, wider than confidence interval
fitted(lm_fit_p) # predicted values
residuals(lm_fit_p) # residuals
anova(lm_fit_p) # anova table
vcov(lm_fit_p) # covariance matrix for model parameters

#visualise
plot(lm_fit_p)

library(dplyr)
#Validating the model with its predictive performance, RMSE
rmse <- function(summarydetail) {
  mean(summarydetail$residuals^2) %>% sqrt()
}

rmse(summ_m) # for math
rmse(summ_p) # for portuguese

#===================== Tree-based method CART (Regression Tree)========================
library(rpart)
library(rpart.plot)			# For Enhanced tree plots

#for math===============
set.seed(2022)

# Continuous Y: Set method = 'anova'
cart_m <- rpart(G3 ~ . - G2 - G1, data = train_m, method = 'anova', control = rpart.control(minsplit = 2, cp = 0))

printcp(cart_m)
## Caution: printcp() shows that if you forgot to change the default CP from 0.01 to 0,
## It would have stopped the tree growing process too early. A lot of further growth at CP < 0.01.

plotcp(cart_m)

print(cart_m)


# 3rd tree is optimal. Choose any CP value betw the 2nd and 3rd tree CP values.
cp1 <- sqrt(1.3683e-01*3.3915e-02)


# [Optional] Extract the Optimal Tree via code instead of eye power ------------
# Compute min CVerror + 1SE in maximal tree cart_m
CVerror.cap <- cart_m$cptable[which.min(cart_m$cptable[,"xerror"]), "xerror"] + cart_m$cptable[which.min(cart_m$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart_m
i <- 1; j<- 4
while (cart_m$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(cart_m$cptable[i,1] * cart_m$cptable[i-1,1]), 1)
# ------------------------------------------------------------------------------

## i = 3 shows that the 3th tree is optimal based on 1 SE rule.
## cp.opt is the same as cp1 (difference due to rounding off error.)


# Prune the max tree using a particular CP value
cart_m_prune <- prune(cart_m, cp = cp1)
printcp(cart_m_prune, digits = 4)
## --- Trainset Error & CV Error --------------------------
## Root node error: 5602/275 = 20.37
## cart_m_prune trainset MSE = 0.7225 * 20.37 = 14.71733
## cart_m_prune CV MSE = 0.7337 * 20.37 = 14.94547

print(cart_m_prune)

rpart.plot(cart_m_prune, nn = T, main = "Optimal Tree for Math")
## The number inside each node represent the mean value of Y.

cart_m_prune$variable.importance
## Weight has the highest importance, disp is second impt.

######### CART for portuguese language====================
set.seed(2022)

# Continuous Y: Set method = 'anova'
cart_p <- rpart(G3 ~ . - G2 - G1, data = train_p, method = 'anova', control = rpart.control(minsplit = 2, cp = 0))

printcp(cart_p)
## Caution: printcp() shows that if you forgot to change the default CP from 0.01 to 0,
## It would have stopped the tree growing process too early. A lot of further growth at CP < 0.01.

plotcp(cart_p)

print(cart_p)

# 3rd tree is optimal. Choose any CP value betw the 2nd and 3rd tree CP values.
cp1 <- sqrt(4.7176e-02*2.8097e-02)


# [Optional] Extract the Optimal Tree via code instead of eye power ------------
# Compute min CVerror + 1SE in maximal tree cart_p
CVerror.cap <- cart_p$cptable[which.min(cart_p$cptable[,"xerror"]), "xerror"] + cart_p$cptable[which.min(cart_p$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree cart_p
i <- 1; j<- 4
while (cart_p$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(cart_p$cptable[i,1] * cart_p$cptable[i-1,1]), 1)
# ------------------------------------------------------------------------------

## i = 3 shows that the 3th tree is optimal based on 1 SE rule.
## cp.opt is the same as cp1 (difference due to rounding off error.)


# Prune the max tree using a particular CP value
cart_p_prune <- prune(cart_p, cp = cp.opt) #based on complexity parameter that is sure optomal
printcp(cart_p_prune, digits = 4)
## --- Trainset Error & CV Error --------------------------
## Root node error: 4734/450 = 10.52
## cart_m_prune trainset MSE = 0.8342 * 10.52 = 8.775784 , RMSE= 2.96
## cart_m_prune CV MSE = 0.8403 * 10.52 = 8.839956, RMSE=2.97

print(cart_p_prune)

rpart.plot(cart_p_prune, nn = T, main = "Optimal Tree for Portuguese")
## The number inside each node represent the mean value of Y.

cart_p_prune$variable.importance
## Weight has the highest importance, disp is second impt.

# prediction for math and portuguese G3 scores =========================
G3hat_m <- predict(cart_m_prune , newdata = test_m)
G3.test_m <- math[!split_m , "G3"]
plot(G3hat_m , G3.test_m)
abline(0, 1)
sqrt(mean((G3hat_m - G3.test_m)^2))

G3hat_p <- predict(cart_p_prune , newdata = test_p)
G3.test_p <- portu[!split_p , "G3"]
plot(G3hat_p , G3.test_p)
abline(0, 1)
mean((G3hat_p - G3.test_p)^2) %>% sqrt()

#======================================================================================
# _______________________
#|Classification Problem |
# _______________________
#R code to import the bank marketing dataset    
bank=read.table("bank.csv",sep=";",header=TRUE) 

names(bank)
dim(bank)
summary(bank)

#observe missing value
library(Amelia)
missmap(bank, main = "Missing values vs observed")
dev.off()

# double checking for missing value
sapply(bank,function(x) sum(is.na(x)))

# We use the skimr package to produce a report on the data
#install.packages('skimr')
library(skimr)
skim(bank)

# Train test split -> 4521*0.664 ~ approximate = 3000
set.seed(1)
train.num <- sample(1:nrow(bank),0.664 * nrow(bank),replace=FALSE)
#subsetting
bank.train <- bank[train.num,]
bank.test <- bank[-train.num, ]
y.test <- bank$y[-train.num]

#==========Logistic Regression================
LR.bank <- glm(y ~.- duration , family=binomial(link='logit'), data=bank.train)
summary(LR.bank)
coef(LR.bank)
summary(LR.bank)$coef
summary(LR.bank)$coef[,4] # p-value for the coef

# type="response" -- output probabilities of the form P(Y = 1|X)
LR.bank.probs = predict(LR.bank , newdata = bank.test, type = "response")
# look at the first 10 values of probability
LR.bank.probs[1:10]

# Test - LR model predictions
LR.bank.predict <- ifelse(LR.bank.probs > 0.5, "yes", "no") # set threshold = 0.5

#Confusion Matrix
table (LR.bank.predict , y.test)

classification_accuracy <- round(mean(LR.bank.predict == y.test),4)
#classification_accuracy <- round( (1319 + 27)/1520, 4 )
classification_accuracy
print(paste('Classification accuracy =',classification_accuracy))
print(paste('Classification error =',1-classification_accuracy))

LR_results_deploy <- data.frame(y.test, LR.bank.predict)
LR_results_deploy

#========================Tree Based Model: Classification Tree ======================
library(rpart)
library(rpart.plot)         # For Enhanced tree plots
set.seed(1)          # For randomisation in 10-fold CV.

# rpart() completes phrase 1 & 2 automatically.
# Change two default settings in rpart: minsplit and cp.
tree.bank <- rpart(y ~ . - duration , data = bank.train, method = 'class',
             control = rpart.control(minsplit = 2, cp = 0)) #cp similar to k in tree library and alpha in the tuning parameter in cv in textbook

# plots the maximal tree and results.
#rpart.plot(tree.bank, nn= T, main = "Maximal Tree in Bank Data")

# prints the maximal tree.bank onto the console.
print(tree.bank)

# prints out the pruning sequence and 10-fold CV errors, as a table.
# (default = 10) randomly folds selection
printcp(tree.bank)

# Display the pruning sequence and 10-fold CV errors, as a chart.
plotcp(tree.bank, main = "Subtrees in Bank Data")

# plotcp uses geometric mean of prune triggers to represent cp on x-axis.
cp1 <- sqrt(0.06213018 * 0.01775148); cp1  # the number on x-axis after the Inf
cp2 <- sqrt(0.01775148 * 0.01035503); cp2  # the next number on x-axis

prune.bank <- prune(tree.bank, cp = cp1)

printcp(prune.bank)

# plots the tree pruned using cp1.
rpart.plot(prune.bank, nn= T, main = "Pruned Tree with Complexity Parameter = 0.033")

prune.bank$variable.importance

# Test CART model prune.bank predictions
tree.pred <- predict(prune.bank , bank.test , type = "class")
table(tree.pred , y.test)
classification_accuracy <- round(mean(tree.pred == y.test),4)
#classification_accuracy <- round( (1323 + 30)/1520, 4 )
print(paste('Classification accuracy =',classification_accuracy))
print(paste('Classification error =',1-classification_accuracy))

Tree_results_deploy <- data.frame(y.test, tree.pred)
Tree_results_deploy

# the end================