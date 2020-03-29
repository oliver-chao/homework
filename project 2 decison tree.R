library(cvTools)
Data = read.table("http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data",sep=",",head=T,row.names=1)
Data$famhistory[Data$famhist=="Present"]=1
Data$famhistory[Data$famhist=="Absent"]=0
Data=Data[-5]
summary(Data)

X = Data[-9]
y = Data[9]

attributeNames = attributes(X)
attributeNames <- as.vector(unlist(attributeNames$names))
attributeNames
N = dim(X)[1]
N

CV <- cvFolds(N, K=10)
CV$TrainSize <- c()
CV$TestSize <- c()

X_train <- X[CV$subsets[CV$which !=1], ];
y_train <- y[CV$subsets[CV$which != 1],];
X_test <- X[CV$subsets[CV$which ==1], ];
y_test <- y[CV$subsets[CV$which ==1],];


(fmla <- as.formula(paste("y_train ~ ", paste(attributeNames, collapse= "+"))))

# Fit logistic regression model to predict the type of wine
w_est = glm(fmla, family=binomial(link="logit"), data=X_train);




# Evaluate the logistic regression for the new data object
p = predict(w_est, newdata=X_test, type="response")
p = ifelse(p>=0.7,1,0)
sum(p == y_test)/length(y_test)

