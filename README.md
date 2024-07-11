# Tesla Stock Price Prediction using RNN & LSTM
## Introduction
This project focuses on predicting Tesla's stock price using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The goal is to leverage historical stock price data to make future price predictions, providing insights and potential investment strategies.

## Dataset
The dataset used in this project is Tesla's stock price data, which includes the following columns:

 - Date
 - Open
 - High
 - Low
 - Close
 - Volume
 - OpenInt
## Project Structure
1. **Data Reading**
 - The dataset is loaded using Pandas.
  ```python

df = pd.read_csv('tsla.us.txt')
```
1. **Data Preprocessing**

 - Select the 'Open' price for the model.
 - Scale the data using StandardScaler for normalization.
   ```python

   train = df.loc[:, ['Open']]
   scaler = StandardScaler()
   train_scaled = scaler.fit_transform(train)
   ```
3. **Feature Engineering**

 - Creating time-step data for training the model.
 - Splitting the data into x_train and y_train.
   ```python

   time_step = 40
   x_train, y_train = [], []
   for i in range(time_step, len(train_scaled)):
      x_train.append(train_scaled[i - time_step:i, 0])
      y_train.append(train_scaled[i, 0])
   ```
4.**Model Building**

  - Building the RNN model using Keras.
   ```python

rnn_model=Sequential()

# Add RNN layer with tanh activation function & input_shape
rnn_model.add(SimpleRNN(128,activation='tanh',return_sequences=True,input_shape=(x_train.shape[1],1)))

# Add Dropout layer to prevent overfitting
rnn_model.add(Dropout(0.20))

# Add RNN layer with tanh activation function
rnn_model.add(SimpleRNN(128,activation='tanh',return_sequences=True))

rnn_model.add(Dropout(0.20))

# Add RNN layer with tanh activation function
rnn_model.add(SimpleRNN(128,activation='tanh',return_sequences=True))


rnn_model.add(SimpleRNN(128))

# Add Dense layer --> output layer
rnn_model.add(Dense(1))
# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')
```
- Building the LSTM model using Keras.
  ```python
     lstm_model=Sequential()

   lstm_model.add(LSTM(64,activation='tanh',return_sequences=True,input_shape=(x_train.shape[1],1)))
 
  lstm_model.add(Dropout(0.20))

  lstm_model.add(LSTM(64,activation='tanh',return_sequences=True))

  lstm_model.add(Dropout(0.20))

  lstm_model.add(SimpleRNN(64,activation='tanh',return_sequences=True))

  lstm_model.add(Dropout(0.20))

  lstm_model.add(LSTM(64))

  lstm_model.add(Dropout(0.20))

  lstm_model.add(Dense(1))
  ```
  
  
5. **Model Training**

 - Training the model with the processed data.
   ```python

   model.fit(x_train, y_train, epochs=20)
   ```
 6. **Prediction and Visualization**

   - Making predictions using the trained model.
   - Visualizing the results with Matplotlib.
                ``` python

                   plt.figure(figsize=(15,8))
                   plt.plot(test_array,color='red',label='Real Price')
                   plt.plot(ypred,color='blue',label='Predict Price')
                   plt.title('Tesla Stock Price')
                   plt.legend()
                   plt.xlabel("Time")
                   plt.ylabel('Price')
                   plt.show()
                                                                                      
                 ```
      **RNN Evaluation**
![Rnn Tesla](https://github.com/Mahmedorabi/Tesla_Stock/assets/105740465/c9975e6d-a70b-4177-b4fc-51bbeab31b8a)

**LSTM Evaluation**

![LSTM tsla](https://github.com/Mahmedorabi/Tesla_Stock/assets/105740465/06c0ec00-260e-4976-a5f5-b03918028ee8)

## Requirements
 - Python 3.x
 - NumPy
 - Pandas
 - Matplotlib
 - Scikit-learn
 - Keras

## Results
The model is trained to predict the opening price of Tesla stocks. The predicted values are visualized alongside the real stock prices to evaluate the model's performance.


 
