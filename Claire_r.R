# setwd("~/Desktop")
# require(data.table)
library(data.table)
getwd()
setwd('C:\\Users\\ADMIN\\Desktop\\Brown_Datathon')


#1
#fread("small_df.csv", sep = ",", header= TRUE)-> data
#2
fread("merge_09_df.csv", sep = ",", header= TRUE)-> data9
#3
# fread("merge_08_df.csv", sep = ",", header= TRUE)-> data8
#4
# fread("merge_07_df.csv", sep = ",", header= TRUE)-> data7
#5
# fread("merge_06_df.csv", sep = ",", header= TRUE)-> data6
#6
# fread("merge_05_df.csv", sep = ",", header= TRUE)-> data5
#7
# fread("merge_04_df.csv", sep = ",", header= TRUE)-> data4


head(data)

library(dplyr)
library(corrplot)
library(car)
library(leaps)
library(mice)
library(caret)
library(kableExtra)
library(lubridate)
library(stringr)
library(data.table)
library(dummy)
library(ShortRead)
library(fastDummies)
library(knitr)
library(randomForest)

#Remove the first column imported from the file
library(dplyr)
data = select(data,-c(1,2))
data = select(data,-c(1))
data = select(data,-c('zip5_demo'))
head(data)
str(data9)

#Check which columns contain NAs.
sapply(data, function(x) sum(is.na
                             (x)))

#Remove the columns with NULL values more than 50% of the total count
names(data)
data_dropped = select(data9,-c(mortgage1_limit,mortgage1_balance,mortgage1_open,mortgage2_limit,
                              mortgage2_balance,mortgage2_open,mortgage3_limit,mortgage3_balance,
                              mortgage3_open, mortgage4_limit,mortgage4_balance,
                              mortgage4_open,mortgage5_limit,
                              mortgage1_loan_to_value,
                              total_revolving_util,
                              total_revolving_trades,
                              total_homeequity_balance,
                              mortgage5_balance,mortgage5_open,
                              homeequity1_limit,homeequity1_balance,homeequity1_open,homeequity2_limit,
                              homeequity2_balance,homeequity2_open,
                              homeequity3_limit,homeequity3_balance,homeequity3_open,
                              homeequity4_limit,homeequity4_balance,homeequity4_open,
                              homeequity5_limit,homeequity5_balance,homeequity5_open,
                              homeequity1_loan_to_value))
summary(data_dropped$bankcard_util)
str(data_dropped)

nrow(data_dropped)

data_dropped = select(data,-c(mortgage1_loan_to_value,
                              total_revolving_util,
                              total_revolving_trades,
                              total_homeequity_balance))
str(data_dropped)
#Check which columns contain NAs to impute them
sapply(data_dropped, function(x) sum(is.na
                                     (x)))
summary(data_dropped$bankcard_util)
str(data_dropped$bankcard_util)
head(data_dropped$bankcard_util,100)


#Set those values that should not be imputed to 0
data_dropped$bankcard_util[is.na
                           (data_dropped$bankcard_util)] <- 0
#data_dropped$total_revolving_util[is.na
(data_dropped$total_revolving_util)] <- 0
data_dropped$total_mortgage_limit[is.na
                                  (data_dropped$total_mortgage_limit)] <- 0
data_dropped$total_mortgage_balance[is.na
                                    (data_dropped$total_mortgage_balance)] <- 0
#data_dropped$mortgage1_loan_to_value[is.na
(data_dropped$mortgage1_loan_to_value)] <- 0
data_dropped$total_homeequity_limit[is.na
                                    (data_dropped$total_homeequity_limit)] <- 0
#data_dropped$total_homeequity_balance[is.na
(data_dropped$total_homeequity_balance)] <- 0

##use MICE to impute these values
##library(caret)
##set.seed(617)
##data_imputed <- predict(preProcess(data_dropped,method='bagImpute'),newdata=data_dropped)

head(data_dropped)

#Data splitting
library(caret)
set.seed(61710)
split = createDataPartition(y=data_dropped$homebuyers,p=0.7,list=F,group = 50)
train = data_dropped[split,]
test = data_dropped[-split,]
nrow(train)
nrow(test)
head(train)
str(train)

#Feature Selectoin
start_mod = lm(homebuyers~1, data = train)
empty_mod = lm(homebuyers~1, data = train)
full_mod = lm(homebuyers~.,data = train)
forwardStepwise = step(start_mod,
                       scope = list(upper=full_mod,lower=empty_mod),
                       direction = 'forward')

summary(forwardStepwise)

start_mod = lm(homebuyers~.,data = train)
empty_mod = lm(homebuyers~1, data = train)
full_mod = lm(homebuyers~.,data = train)
backwardStepwise = step(start_mod,
                        scope = list(upper=full_mod,lower=empty_mod),
                        direction = 'backward')

summary(backwardStepwise)

##hybrid:
start_mod = lm(homebuyers~1, data = train)
empty_mod = lm(homebuyers~1, data = train)
full_mod = lm(homebuyers~.,data = train)
hybridStepwise = step(start_mod,
                      scope = list(upper=full_mod,lower=empty_mod),
                      direction = 'both')

summary(hybridStepwise)

#Model fitting
##all data:
set.seed(617)
model_LR_1 = lm(homebuyers ~.,train)
pred_LR_1 = predict(model_LR_1, newdata = test)
rmse_LR_test1 = sqrt(mean((pred_LR_1-test$homebuyers)^2));rmse_LR_test1
mean((pred_LR_1-test$homebuyers)^2)
plot(model_LR_1)
summary(model_LR_1)

#Loss 1:
mean((pred_LR_1-test$homebuyers)^2)

#Loss 2:
differences = pred_LR_1-test$homebuyers
under = as.numeric(ifelse(differences < 0 , differences, 0))
over = as.numeric(ifelse(differences > 0 , differences, 0))
under_square_scaled = under*under*10
over_square = over*over
square_sum = mean(under_square_scaled+over_square);square_sum

##forward model fitting:
set.seed(617)
model_LR_2 = lm(homebuyers ~ first_homebuyers + mortgage_open + household_count +
                  bankcard_open + autoloan_open + bankcard_limit + age + total_homeequity_trades +
                  bankcard_trades + bankcard_balance + zip5_sep + studentloan_open,train)
pred_LR_2 = predict(model_LR_2, newdata = test)
rmse_LR_test2 = sqrt(mean((pred_LR_2-test$homebuyers)^2));rmse_LR_test2
mean((pred_LR_2-test$homebuyers)^2) #use this to compare to benchmark.

##backward model fitting:
set.seed(617)
model_LR_3 = lm(homebuyers ~ zip5_sep + bankcard_limit + bankcard_balance +
                  bankcard_trades + total_homeequity_trades + autoloan_open +
                  studentloan_open + bankcard_open + mortgage_open + age +
                  household_count + first_homebuyers,train)
pred_LR_3 = predict(model_LR_3, newdata = test)
rmse_LR_test3 = sqrt(mean((pred_LR_3-test$homebuyers)^2));rmse_LR_test3


#merge datasets
str(data9) #26
data9 = select(data9,-c(mortgage1_limit,mortgage1_balance,mortgage1_open,mortgage2_limit,
                        mortgage2_balance,mortgage2_open,mortgage3_limit,mortgage3_balance,
                        mortgage3_open, mortgage4_limit,mortgage4_balance,
                        mortgage4_open,mortgage5_limit,
                        mortgage1_loan_to_value,
                        mortgage5_balance,mortgage5_open,
                        homeequity1_limit,homeequity1_balance,homeequity1_open,homeequity2_limit,
                        homeequity2_balance,homeequity2_open,
                        homeequity3_limit,homeequity3_balance,homeequity3_open,
                        homeequity4_limit,homeequity4_balance,homeequity4_open,
                        homeequity5_limit,homeequity5_balance,homeequity5_open,
                        homeequity1_loan_to_value))
data9 = select(data9,-c('V1','zip5_demo'))
str(data9)
data8 = select(data8,-c(mortgage1_loan_to_value))
data7 = select(data7,-c(mortgage1_loan_to_value))
data6 = select(data6,-c(mortgage1_loan_to_value))
data5 = select(data5,-c(mortgage1_loan_to_value))
data4 = select(data4,-c(mortgage1_loan_to_value))

data8 = select(data8,-c(homebuyers))
data7 = select(data7,-c(homebuyers))
data6 = select(data6,-c(homebuyers))
data5 = select(data5,-c(homebuyers))
data4 = select(data4,-c(homebuyers))

str(data9)
str(data8)
head(data8)

cbind(df1,df2)



colnames(data9) <- paste(colnames(data9), "9", sep = "_")
colnames(data8) <- paste(colnames(data8), "8", sep = "_")
colnames(data7) <- paste(colnames(data7), "7", sep = "_")
colnames(data6) <- paste(colnames(data6), "6", sep = "_")
colnames(data5) <- paste(colnames(data5), "5", sep = "_")
colnames(data4) <- paste(colnames(data4), "4", sep = "_")
total<- cbind(data9,data8,data7,data6,data5,data4)
str(total)

total <- merge(data9,data8,by="zip9_code")
total <- merge(total,data7,by="zip9_code")
head(total)
str(total)
names(total)

total = select(total,-c(zip9_code_8,zip9_code_7,zip9_code_6,zip9_code_5,zip9_code_4))
names(total)

write.csv(total,"C:\\Users\\clairecheng\\Desktop\\Total_data.csv", row.names = TRUE)

names(data_dropped)
str(data_dropped)
str(data9)

getwd()

write.csv(data9,"C:\\Users\\ADMIN\\Desktop\\Brown_Datathon\\Total_data.csv", row.names = TRUE)




####################################################
sapply(total, function(x) sum(is.na
                              (x)))

#x
total$bankcard_util.x[is.na
                      (total$bankcard_util.x)] <- 0
total$total_revolving_util.x[is.na
                             (total$total_revolving_util.x)] <- 0
total$total_mortgage_limit.x[is.na
                             (total$total_mortgage_limit.x)] <- 0
total$total_mortgage_balance.x[is.na
                               (total$total_mortgage_balance.x)] <- 0
total$mortgage1_loan_to_value.x[is.na
                                (total$mortgage1_loan_to_value.x)] <- 0
total$total_homeequity_limit.x[is.na
                               (total$total_homeequity_limit.x)] <- 0
total$total_homeequity_balance.x[is.na
                                 (total$total_homeequity_balance.x)] <- 0

#y
total$bankcard_util.y[is.na
                      (total$bankcard_util.y)] <- 0
total$total_revolving_util.y[is.na
                             (total$total_revolving_util.y)] <- 0
total$total_mortgage_limit.y[is.na
                             (total$total_mortgage_limit.y)] <- 0
total$total_mortgage_balance.y[is.na
                               (total$total_mortgage_balance.y)] <- 0
total$mortgage1_loan_to_value.y[is.na
                                (total$mortgage1_loan_to_value.y)] <- 0
total$total_homeequity_limit.y[is.na
                               (total$total_homeequity_limit.y)] <- 0
total$total_homeequity_balance.y[is.na
                                 (total$total_homeequity_balance.y)] <- 0

#Data splitting
total = select(total,-c('homebuyers.y'))
library(caret)
set.seed(61710)
split = createDataPartition(y=total$homebuyers.x,p=0.7,list=F,group = 50)
train = total[split,]
test = total[-split,]
nrow(train)
nrow(test)
head(train)
str(train)
str(total)

summary(total$homebuyers.x)
summary(total$homebuyers.y)

#Model fitting
##all data:
set.seed(617)
model_LR_1 = lm(homebuyers.x ~.,train)
pred_LR_1 = predict(model_LR_1, newdata = test)
rmse_LR_test1 = sqrt(mean((pred_LR_1-test$homebuyers.x)^2));rmse_LR_test1
mean((pred_LR_1-test$homebuyers.x)^2)
plot(model_LR_1)
summary(model_LR_1)

#Loss 1:
mean((pred_LR_1-test$homebuyers.x)^2)

#Loss 2:
differences = pred_LR_1-test$homebuyers.x
under = as.numeric(ifelse(differences < 0 , differences, 0))
over = as.numeric(ifelse(differences > 0 , differences, 0))
under_square_scaled = under*under*10
over_square = over*over
square_sum = mean(under_square_scaled+over_square);square_sum