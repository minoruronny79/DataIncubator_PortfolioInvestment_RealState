remove(list = ls())

library(PortfolioAnalytics)
library(xts)
library(dplyr)
library(ggplot2)
library(tseries)
library(Amelia)
library(dplyr)

#Retrieving dataset
port<-read.csv("F:/PythonTraining/Python_CaseStudy/PCS_DataIncubator_PriceHouses_GDP/top10.csv", sep = ",")
colnames(port)
str(port)


#Transforming the variable as Date format
port2<-port
port2$fecha<-as.Date(port2$Date)
row.names(port2) <- port2[,"fecha"]  #Setting the index
port2<-select(port2, -Date, -fecha)

port2


#############################################################
############Optimizing a portfolio###########################
#############################################################
colnames(port2)       

portafolio1<-portfolio.spec(assets = c("ZHVI_2bedroom_Nevada", 
                                       "ZHVI_3bedroom_Nevada",
                                       "ZHVI_3bedroom_DistrictofColumbia",
                                       "ZHVI_2bedroom_Colorado",
                                       "ZHVI_1bedroom_California",
                                       "ZHVI_1bedroom_Florida",
                                       "ZHVI_1bedroom_Colorado", 
                                       "ZHVI_2bedroom_California",
                                       "ZHVI_4bedroom_Nevada",
                                       "ZHVI_3bedroom_Colorado"
                                       ))
                           
portafolio1<-add.constraint(portfolio=portafolio1, type="weight_sum",
                            min_sum=0.99, max_sum=1.01)

portafolio1 <- add.constraint(portfolio=portafolio1, type="box", min=0, max=0.5)

portafolio1<-add.objective(portfolio = portafolio1, type="return",
                           name="mean")


portafolio1

opt<-optimize.portfolio(R = port2, portfolio = portafolio1, 
                        optimize_method = "random", trace = TRUE)


opt


