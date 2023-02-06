import pandas as pd
import numpy as np
from category_encoders import OneHotEncoder
import os


def two_month_source(data):
    #筛选出11月和12月的数据
    key = 0
    for item in data:
        key +=1
        if item[0] == 2022 and item[1] == 11 and item[2] == 2:
            break
    data = data[key-1:]
    key = 0
    for item in data:
        key +=1
        if item[0] == 2023 and item[1] == 1 and item[2] == 1:
            break
    data = data[:key-1]
    data_source = data.copy()
    return data_source
    
def source_process(data_source):
    data_source['RATE'] = data_source['SYBW']/data_source['BWZS']
    date = np.array(data_source['FBSJ'])
    rate = np.array(data_source['RATE'])
    count=0
    sum=0
    data = []
    last = []
    for i in range(0,data_source.shape[0]):
        temp = date[i]
        y=int(temp[0:4])
        m=int(temp[5:7])
        d=int(temp[8:10])
        h=int(temp[11:13])
        mi=int(temp[14:16])
        if mi<20 and mi >=0:
            mi = 1
            if i!=0 and [y,m,d,h,mi] != last:
                data.append([last[0],last[1],last[2],last[3],last[4],sum/count])
                sum = 0
                count = 0
            last = [y,m,d,h,mi]
            sum+=rate[i]
            count+=1
            continue
        if mi<40 and mi >=20:
            mi = 2
            if i!=0 and [y,m,d,h,mi] != last:
                data.append([last[0],last[1],last[2],last[3],last[4],sum/count])
                sum = 0
                count = 0
            last = [y,m,d,h,mi]
            sum+=rate[i]
            count+=1
            continue
        if mi<60 and mi >=40:
            mi = 3
            if i!=0 and [y,m,d,h,mi] != last:
                data.append([last[0],last[1],last[2],last[3],last[4],sum/count])
                sum = 0
                count = 0
            last = [y,m,d,h,mi]
            sum+=rate[i]
            count+=1
            continue
    data = two_month_source(data)
    print("生成了",len(data),"条源数据")

    return data

def get_infl_elem(data):
    key = 0
    for d in data['rq']:
        key +=1
        if str(d) == '2022-11-02':
            break
    data = data[key:]
    key = 0
    for d in data['rq']:
        key +=1
        if str(d) == '2023-01-01':
            break
    data_2 = data[:key-1]
    weather = []
    count = 0
    for i in data_2['weather']:
        if i not in weather:
            count +=1
            weather.append(i)
    print("天气的种类：",weather)
    #data_2缺失

    data_2 = np.array(data_2)

    weather = []
    weeks = []
    for i in range(0,data_2.shape[0]):
        time = int(data_2[i,1][3:])
        wea = data_2[i,3]
        wee = data_2[i,2]
        if i != 0:
            if (last + 10)%60 != time:
                weather.append(wea)
                weeks.append(wee)
        weather.append(wea)
        weeks.append(wee)
        last=time
    weather.append(wea)#缺少一个排查不出来
    weeks.append(wee)

    return weather,weeks

def train_test(data,all_elem = True):
    """"
    lookback = int,为回溯的天数
    all_elem = True,代表把星期也作为影响要素
    """
    lookback = 504#回溯时间
    forecast_long = 3*24
    period_long = forecast_long*7
    inf_size = 5 #天气个数

    if all_elem == True:
        inf_size = 12
    
    x_train=[]
    y_train=[]
    x_train_influence=[]
    for i in range(4320-lookback-forecast_long-period_long):
        x_train.append(data[i:i+lookback])
        x_train_influence.append(data[i+lookback:i+lookback+forecast_long,1:inf_size+1])
        y_train.append(data[i+lookback:i+lookback+forecast_long,0:1])
    x_train=np.array(x_train)
    x_train_influence=np.array(x_train_influence)
    y_train=np.array(y_train).reshape(4320-lookback-forecast_long-period_long,forecast_long)
    x_test = []
    x_test_influence = []
    y_test = []
    for i in range(4320-lookback-forecast_long-period_long,4320-lookback-forecast_long,72):
        x_test.append(data[i:i+lookback])
        x_test_influence.append(data[i+lookback:i+lookback+forecast_long,1:inf_size+1])
        y_test.append(data[i+lookback:i+lookback+forecast_long,0])
    x_test = np.array(x_test)
    x_test_influence = np.array(x_test_influence)
    y_test = np.array(y_test).reshape(504)

    return (x_train,x_train_influence,y_train),(x_test,x_test_influence,y_test)

def Data_process(name,all_elem = True):
    name = name +'.csv'
    file_sourse = os.path.join('dataset/datasourse',name)
    file_weather = os.path.join('dataset/dataweather',name)
    data_source = pd.read_csv(file_sourse).iloc[::-1]
    data_weather = pd.read_csv(file_weather).iloc[::-1]
    del data_weather['parkno']
    del data_weather['parkname']

    data_source = source_process(data_source)
    weather,weeks = get_infl_elem(data_weather)

    for i in range(len(data_source)):#增加影响维度
        data_source[i].append(weather[i*2])
        data_source[i].append(weeks[i*2])

    data = pd.DataFrame(data_source)
    #独热编码
    encoder = OneHotEncoder(cols=[6,7], 
                        use_cat_names=True).fit(data) 
    encoder_data = encoder.transform(data)

    #删除无关要素
    del encoder_data[0]
    del encoder_data[1]
    del encoder_data[2]
    del encoder_data[3]
    del encoder_data[4]
    data = np.array(encoder_data)
    train_data,test_data = train_test(data,all_elem)

    return train_data,test_data






    



