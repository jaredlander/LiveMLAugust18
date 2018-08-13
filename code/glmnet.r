library(glmnet)
library(useful)
library(coefplot)
library(magrittr)

land_train <- readr::read_csv('data/manhattan_Train.csv')
land_test <- readRDS('data/manhattan_Test.rds')

View(land_train)

valueFormula <- TotalValue ~ FireService + 
    ZoneDist1 + ZoneDist2 + Class + LandUse + 
    OwnerType + LotArea + BldgArea + ComArea + 
    ResArea + OfficeArea + RetailArea + 
    GarageArea + FactryArea + NumBldgs + 
    NumFloors + UnitsRes + UnitsTotal + 
    LotFront + LotDepth + BldgFront + 
    BldgDepth + LotType + Landmark + BuiltFAR +
    Built + HistoricDistrict - 1

valueFormula
class(valueFormula)

value1 <- lm(valueFormula, data=land_train)
coefplot(value1, sort='magnitude')

landX_train <- build.x(valueFormula, data=land_train, contrasts=FALSE, sparse=TRUE)
landX_train
landY_train <- build.y(valueFormula, data=land_train)
head(landY_train, n=20)

value2 <- glmnet(x=landX_train, y=landY_train, family='gaussian')
coefpath(value2)

animation::cv.ani(k=10)

value3 <- cv.glmnet(x=landX_train, y=landY_train, family='gaussian', nfolds=5)
coefpath(value3)
plot(value3)

coefplot(value3, lambda='lambda.min', sort='magnitude')
coefplot(value3, lambda='lambda.1se', sort='magnitude')
coefplot(value3, lambda='lambda.1se', sort='magnitude') + xlim(-1000, 1000)

value4 <- cv.glmnet(x=landX_train, y=landY_train, family='gaussian',
                    nfolds=5, alpha=1)
coefplot(value4, lambda='lambda.1se', sort='magnitude')

value5 <- cv.glmnet(x=landX_train, y=landY_train, family='gaussian',
                    nfolds=5, alpha=0)
coefpath(value5)
plot(value5)
coefplot(value5, lambda='lambda.1se', sort='magnitude')

value6 <- cv.glmnet(x=landX_train, y=landY_train, family='gaussian',
                    nfolds=5, alpha=0.6)
coefpath(value6)
plot(value6)
coefplot(value6, lambda='lambda.1se', sort='magnitude')

landX_test <- build.x(valueFormula, data=land_test, contrasts=FALSE, sparse=TRUE)

landPredictions6 <- predict(value6, newx=landX_test, s='lambda.1se')
head(landPredictions6, n=10)
