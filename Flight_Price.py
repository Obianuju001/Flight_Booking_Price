import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Clean_Dataset.csv')
# data.head()

df = data.copy()

# df.drop('Unnamed: 0', axis = 1, inplace = True)
# df.drop('flight', axis = 1, inplace = True)

# df.dropna(inplace = True)
# df.isnull().sum()

categorical = df.select_dtypes(exclude = 'number')
numerical = df.select_dtypes(include = 'number')

# print(f"\t\t\t\t\tCategorical data")
# display(categorical.head(3))

# print(f"\n\n\t\t\t\t\tNumerical data")
# display(numerical.head(3))

# PREPROCESSSING
# # Standardization
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# scaler = StandardScaler()
# encoder = LabelEncoder()

# for i in numerical.columns: # ................................................. Select all numerical columns
#     if i in df.drop('price', axis = 1).columns: # ...................................................... If the selected column is found in the general dataframe
#         df[i] = scaler.fit_transform(df[[i]]) # ................................ Scale it

# # for i in categorical.columns: # ............................................... Select all categorical columns
#     if i in df.drop('price', axis = 1).columns: # ...................................................... If the selected columns are found in the general dataframe
#         df[i] = encoder.fit_transform(df[i])# .................................. encode it

# y = df['price']
# x = df.drop('price', axis = 1)

# df.head()

# # Assumption of Multicolinearity
# plt.figure(figsize = (9, 3))
# sns.heatmap(df.corr(), annot = True, cmap = 'BuPu')

# Train And Test Split
x = df.drop('price', axis = 1)
y = df.price

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size = 0.80, random_state = 69)
# # print(f'xtrain: {xtrain.shape}')
# # print(f'ytrain: {ytrain.shape}')
# # print(f'xtest: {xtest.shape}')
# # print(f'ytest: {ytest.shape}')

train_set = pd.concat([xtrain, ytrain], axis = 1)
test_set = pd.concat([xtest, ytest], axis = 1)

# # print(f'\t\tTrain DataSet')
# # display(train_set.head())
# # print(f'\n\t\tTest DataSet')
# # display(test_set.head())

# # --------- Modelling ----------
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, mean_squared_error

# lin_reg = LinearRegression()
# lin_reg.fit(xtrain, ytrain) # --------------------------Create a linear regression model

# # -------------- cross validation -------------
# cross_validate = lin_reg.predict(xtrain)
# score = r2_score(cross_validate, ytrain)
# # print(f'The Cross Validation Score is: {score.round(2)}')

# # Model Metrics and Testing
# test_prediction = lin_reg.predict(xtest)
# score = r2_score(test_prediction, ytest)
# # # print(f'The Cross Validation Score is: {score.round(2)}')

# # save model
# # model = pickle.dump(lin_reg, open('Model.pkl', 'wb'))
# # print('\nModel is saved\n')


#-----------------------STREAMLIT DEVELOPMENT----------------------------------

model = pickle.load(open('Model.pkl','rb'))

st.markdown("<h1 style = 'color: 00A9FF; text-align: center;font-family: montserrat, Helvetica, sans-serif; '>FLIGHT BOOKING PRICES</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: #45474B; text-align: center;font-family: script, Helvetica, sans-serif; '> Flight Prices</h3>", unsafe_allow_html= True)
st.image('Flight.png', width = 600)

password = ['one', 'two', 'three']
username = st.text_input('Pls enter your username')
passes = st.text_input('Pls input password')

if passes in password:
    st.toast('Registered User')
    print(f'Welcome {username}, Pls enjoy your usage as a registered user')

    st.markdown("<h2 style = 'color: #0F0F0F; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF STUDY </h2>", unsafe_allow_html= True)

    st.markdown('<br2>', unsafe_allow_html= True)

    st.markdown("<p>The aviation industry plays a crucial role in global transportation, connecting people and businesses across the world. As travelers increasingly rely on air travel for both leisure and business, understanding and predicting flight prices have become essential for optimizing travel plans and making informed decisions. In this project, we focus on the task of predicting flight prices, utilizing data-driven approaches for more accurate and reliable estimates.</p>",unsafe_allow_html= True)


    st.sidebar.image('user.png')

    dx = df[['airline', 'source_city', 'departure_time', 'stops', 'arrival_time',
        'destination_city', 'class', 'duration', 'days_left']]
    dx.rename(columns = {'class': 'Class_cat'} )

    st.write(dx.head())

    airline = st.sidebar.selectbox("airline", dx['airline'].unique())
    source_city = st.sidebar.selectbox("source_city", dx['source_city'].unique())
    departure_time = st.sidebar.selectbox("departure_time", dx['departure_time'].unique())
    stops = st.sidebar.selectbox("stops", dx['stops'].unique())
    arrival_time = st.sidebar.selectbox("arrival_time", dx['arrival_time'].unique())
    destination_city = st.sidebar.selectbox('destination_city', dx['destination_city'].unique())
    Class_cat = st.sidebar.selectbox("class", dx['class'].unique())
    duration = st.sidebar.selectbox("duration", dx['duration'].unique())
    days_left = st.sidebar.selectbox("days_left", dx['days_left'].unique())       


    # Bring all the inputs into a dataframe
    input_variable = pd.DataFrame([{
        'airline': airline,
        'source_city': source_city,
        'departure_time': departure_time,
        'stops': stops,
        'arrival_time': arrival_time,
        'destination_city': destination_city,
        'class': Class_cat,
        'duration': duration,
        'days_left': days_left, 
    
    }])

    # Reshape the Series to a DataFrame
    # input_variable = input_data.to_frame().T

    # Model Metrics and Testing
    # test_prediction = model.predict(xtest)
    # score = r2_score(test_prediction, ytest)

    st.write(input_variable)
    cat = input_variable.select_dtypes(include = ['object', 'category'])
    num = input_variable.select_dtypes(include = 'number')

        # Standard Scale the Input Variable.
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    for i in input_variable.columns:
        if i in num.columns:
            input_variable[i] = scaler.fit_transform(input_variable[[i]])
    for i in input_variable.columns:
        if i in cat.columns: 
            input_variable[i] = LabelEncoder().fit_transform(input_variable[i])


    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("<h2 style = 'color: #0A2647; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)


    if st.button('Press To Predict'):
        predicted = model.predict(input_variable)
        st.toast('Flight Price Predicted')
        st.image('Prediction_tick.png', width = 100)
        st.success(f'Price of Flight is {predicted}')
else:
    st.error('You are not a registered user. But you have three trials')

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h8>FLIGHT PRICE built by OBIANUJU ONYEKWELU</h8>", unsafe_allow_html=True)

