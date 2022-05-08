library(forecast)
library(lubridate)
library(fpp)
library(xts)

data_litecoin<-read.csv("/Users/sabrinachowdhury/Desktop/BigData_Spring2022/Project/Data_Set/coin_Litecoin.csv")
data_litecoin$avg_price <- rowMeans(data_litecoin[,5:6], na.rm=TRUE,dims=1)

data_xts <- xts(x=data_litecoin$avg_price, order.by=as.Date(data_litecoin$Date,format =c("%Y-%m-%d %H:%M:%S")),frequency=365)

m_ets = ets(data_xts) #fit exp smooth model
# accuracy(m_ets) #check accuracy
# f_ets = forecast(m_ets, h=730, level = 51, robust=TRUE, model='bats', allow.multiplicative.trend=TRUE) # forecast 12 months into the future
# plot(f_ets,main="litecoin prediction using ETS",ylab="litecoin price (per unit) in $",xlim=c(2992,3721))
# #plot(f_ets,main="litecoin prediction using ETS",ylab="litecoin price (per unit) in $", xlim=c(2018.9,2019.9), ylim=c(0,1000))
# write.csv(f_ets,"/Users/sabrinachowdhury/Desktop/BigData_Spring2022/Project/Data_Set/ets_Litecoin_forecast.csv")
# 
# 
# # arima model
arimafit <- auto.arima(data_xts)
accuracy(arimafit) #check accuracy
fcast<-forecast(arimafit,h=730)
plot(fcast,main="litecoin prediction using ARIMA",ylab="litecoin price (per unit) in $")# xlim=c(2018.9,2019.9), ylim=c(0,1000))
write.csv(fcast$upper,"/Users/sabrinachowdhury/Desktop/BigData_Spring2022/Project/Data_Set/Arima_ELitecoin_forecast.csv")



#############################
#### Neural network###########

# library(keras)
# library(tm)
# library(ggplot2)
# 
# 
# 
# 
# #mnth <- list(month.abb, unique(floor(time(data_xts))))
# #creating data frame
# data <- as.data.frame(t(matrix(data_xts, 729)))
# 
# #Creating training data, training labels and test data
# ind = sample(2,nrow(data),replace = T,prob = c(0.5,0.5))
# training = data[ind==1,1:(ncol(data)-1)]
# #training
# test = data[ind==2,1:(ncol(data)-1)]
# traintarget = data[ind==1,ncol(data)]
# testtarget = data[ind==2,ncol(data)]
# 
# # one hot encoding labels
# trainLabels = to_categorical(traintarget)
# testLabels = to_categorical(testtarget)
# #print(testLabels)
# 
# dim(training)
# dim(trainLabels)
# 
# # create neural network
# 
# model = keras_model_sequential()
# 
# # model %>%
# #   layer_dense(units = 32, input_shape = c(784)) %>%
# #   layer_activation('relu') %>%
# #   layer_dense(units = 10) %>%
# #   layer_activation('softmax')%>%
# #   layer_dense(units=dim(trainLabels)[2])
# layer_flatten(model)
# model %>%
#   layer_dense(units=dim(training)[2],
#               activation='softmax',
#               input_shape=dim(training)[2])  %>%
#   #              input_shape=c(11))  %>%
#   layer_dense(units=dim(training)[2],
#               activation='softmax')  %>%
#   layer_dense(units=dim(training)[2],
#               activation='softmax')  %>%
#   layer_dense(units=dim(training)[2],
#               activation='softmax')  %>%
#   layer_dense(units=dim(training)[2],
#               activation='softmax')  %>%
#   layer_dense(units=dim(trainLabels)[2])
# #  layer_dense(units=1)
# 
# summary(model)
# 
# # compile model
# model %>%
#   compile(loss='mse',#'categorical_crossentropy',
#           optimizer='adam',
#           metrics='accuracy')
# 
# summary(model)
# 
# # train model
# history = model %>%
#   fit(as.matrix(training),
#       trainLabels,
#       #      traintarget,
#       epoch=400,
#       batch_size=1)
# 
# # make predictions
# avg_bit = mean(data_litecoin$High)
# btc_krs_pred<-abs(model %>% predict(as.matrix(test)))*(test/min(test))*avg_bit*avg_bit
# count<-c(1:365)
# #count<-c(1:24)
# #btc_p<-c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
# btc_p<-rep(0,365)
# cnt <- 1
# for (val in count) {
#   btc_p[cnt] = btc_krs_pred[1,cnt]
#   cnt = cnt+1
#   
# }
# 
# 
# plot(count,btc_p,type="l",
#      col="blue",main="Bitcoin prediction using Neural Network",ylab="Bitcoin price (per unit) in $",)
# 
# 
