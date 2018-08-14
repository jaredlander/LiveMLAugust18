library(useful)
library(xgboost)
library(coefplot)
library(magrittr)
library(dygraphs)

land_train <- readr::read_csv('data/manhattan_Train.csv')
land_test <- readRDS('data/manhattan_Test.rds')
land_val <- readRDS('data/manhattan_Validate.rds')

table(land_train$HistoricDistrict)

histFormula <- HistoricDistrict ~ FireService + 
    ZoneDist1 + ZoneDist2 + Class + LandUse + 
    OwnerType + LotArea + BldgArea + ComArea + 
    ResArea + OfficeArea + RetailArea + 
    GarageArea + FactryArea + NumBldgs + 
    NumFloors + UnitsRes + UnitsTotal + 
    LotFront + LotDepth + BldgFront + 
    BldgDepth + LotType + Landmark + BuiltFAR +
    Built + TotalValue - 1

landX_train <- build.x(histFormula, data=land_train, contrasts=FALSE, sparse=TRUE)
landY_train <- build.y(histFormula, data=land_train) %>% as.factor() %>% as.integer() - 1
head(landY_train, n=20)

landX_val <- build.x(histFormula, data=land_val, contrasts=FALSE, sparse=TRUE)
landY_val <- build.y(histFormula, data=land_val) %>% as.factor() %>% as.integer() - 1

landX_test <- build.x(histFormula, data=land_test, contrasts=FALSE, sparse=TRUE)
landY_test <- build.y(histFormula, data=land_test) %>% as.factor() %>% as.integer() - 1

xgTrain <- xgb.DMatrix(data=landX_train, label=landY_train)
xgVal <- xgb.DMatrix(data=landX_val, label=landY_val)

xg1 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=1
)
xg1

xgb.plot.multi.trees(xg1, feature_names=colnames(landX_train))


xg2 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=1,
    eval_metric='logloss',
    watchlist=list(train=xgTrain)
)

xg3 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=100,
    eval_metric='logloss',
    watchlist=list(train=xgTrain),
    print_every_n=1
)

xg4 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=300,
    eval_metric='logloss',
    watchlist=list(train=xgTrain),
    print_every_n=1
)

xg5 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=500,
    eval_metric='logloss',
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=1
)

xg5$evaluation_log
dygraph(xg5$evaluation_log)

xg6 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=500,
    eval_metric='logloss',
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=1,
    early_stopping_rounds=70
)
dygraph(xg6$evaluation_log)
xg6$best_iteration
xg6$best_score

xgb.plot.multi.trees(xg6, feature_names=colnames(landX_train))

xg6 %>% 
    xgb.importance(feature_names=colnames(landX_train)) %>% 
    xgb.plot.importance()


xg7 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=1000,
    eval_metric='logloss',
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=1,
    early_stopping_rounds=70,
    max_depth=8
)

xg8 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=1000,
    eval_metric='logloss',
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=1,
    early_stopping_rounds=70,
    max_depth=3
)

xg9 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=2500,
    eval_metric='logloss',
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1
)

xg10 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=2500,
    eval_metric='logloss',
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1,
    subsample=0.8
)

xg11 <- xgb.train(
    data=xgTrain,
    objective="binary:logistic",
    nrounds=2500,
    eval_metric='logloss',
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1,
    subsample=0.7, colsample_bytree=0.5
)

