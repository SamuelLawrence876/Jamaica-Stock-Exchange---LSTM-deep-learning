import time
import numpy as np
import pandas as pd
import streamlit as st
from keras.layers import Bidirectional
from keras.layers import ConvLSTM2D
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from numpy import array
from tensorflow.keras.layers import Flatten


def main():
    """Sentiment Analysis Emoji App """

    st.title("Jamaica Stock Exchange LSTM prediction")

    st.subheader('About LSTM:')
    st.write('Long Short-Term Memory networks, or LSTMs for short, can be applied to time series forecasting. '
             'There are many types of LSTM models that can be used for each specific type of time series '
             'forecasting problem. In this scenario we are using Univariate LSTM Models based on linear data '
             'Though this method is not ideal for investment purposes it is interesting to see it put into practice')

    st.markdown(
        "**Acknowledgments:** I'd like to thank Jason Brownlee for his contribution of algorithms and explanation towards this "
        "project. His work relating to LSTM sequence predictions can be found at: "
        "https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/   ")

    LSTM_T = ["Vanilla LSTM", "Bidirectional LSTM", "CNN LSTM", "Conv LSTM", "Multilayer Perceptron Regression"]
    choice = st.sidebar.selectbox("vanilla LSTM", LSTM_T)
    # if choice == 'Multilayer Perceptron Regression':

    if choice == 'Vanilla LSTM':
        url = 'https://www.jamstockex.com/market-data/download-data/index-history/main-market/JSE-Index/2010-08-08/2020-08-10'
        df = pd.read_html(url)

        jse = df[0]

        # JSE PLot
        # plt.figure(figsize=(18, 18))
        # plt.plot(jse['Value'])

        # New DF
        jse_value = jse['Value']
        raw_seq = jse_value
        st.subheader("Jamaica stock Exchange graph:")

        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        # choose a number of time steps
        n_steps = 3
        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        # define model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=200, verbose=0)

        # demonstrate prediction
        x_input = array(jse_value.tail(3).tolist())
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)

        # Charts

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = np.random.randn(1, 1)
        chart = st.line_chart(jse_value, use_container_width=True)

        for i in range(1, 101):
            new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)
            # chart.add_rows(new_rows)
            progress_bar.progress(i)
            # last_rows = new_rows
            time.sleep(0.05)

        progress_bar.empty()

        # rerun.
        st.button("Re-run")

        st.subheader('About Vanilla LSTM:')
        st.write(' Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units, and an output layer '
                 'used to make a prediction. ')

        st.write('The LSTM model has predicted that the next value in the sequence will be ' + str(yhat[0][0]))
        last_price = jse_value.iloc[-1]
        st.write('The current jamaica stock exchange index value is ' + str(last_price))

        difference = yhat[0][0] - jse_value.iloc[-1]

        st.write('This reflects a difference of ' + str(round(difference, 2)))

    if choice == 'Bidirectional LSTM':
        st.subheader("Bidirectional LSTM")
        url = 'https://www.jamstockex.com/market-data/download-data/index-history/main-market/JSE-Index/2010-08-08/2021-01-01'
        df = pd.read_html(url)

        jse = df[0]

        # New DF
        jse_value = jse['Value']

        # split a univariate sequence into samples
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        raw_seq = jse_value
        # choose a number of time steps
        n_steps = 3
        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        # define model
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=200, verbose=0)

        # demonstrate prediction
        x_input = array(jse_value.tail(3).tolist())
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = np.random.randn(1, 1)
        chart = st.line_chart(jse_value, use_container_width=True)

        for i in range(1, 101):
            new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)
            # chart.add_rows(new_rows)
            progress_bar.progress(i)
            # last_rows = new_rows
            time.sleep(0.05)

        progress_bar.empty()

        st.subheader('About LSTM:')
        st.write(
            'We can implement a Bidirectional LSTM for univariate time series forecasting by wrapping the first '
            'hidden layer in a wrapper layer called Bidirectional. This model learns the input sequence both forward '
            'and backwards and concatenate both interpretations')

        st.write('The LSTM model has predicted that the next value in the sequence will be ' + str(yhat[0][0]))
        last_price = jse_value.iloc[-1]
        st.write('The current jamaica stock exchange index value is ' + str(last_price))

        difference = yhat[0][0] - jse_value.iloc[-1]

        st.write('This reflects a difference of ' + str(round(difference, 2)))

    if choice == 'CNN LSTM':
        st.subheader("About CNN LSTM")
        url = 'https://www.jamstockex.com/market-data/download-data/index-history/main-market/JSE-Index/2010-08-08/2020-08-10'
        df = pd.read_html(url)

        jse = df[0]

        # New DF
        jse_value = jse['Value']
        raw_seq = jse_value

        # split a univariate sequence into samples
        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        n_steps = 4
        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
        n_features = 1
        n_seq = 2
        n_steps = 2
        X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                  input_shape=(None, n_steps, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=500, verbose=0)

        # demonstrate prediction
        x_input = array(jse_value.tail(4).tolist())
        x_input = x_input.reshape((1, n_seq, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = np.random.randn(1, 1)
        chart = st.line_chart(jse_value, use_container_width=True)

        for i in range(1, 101):
            new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)
            # chart.add_rows(new_rows)
            progress_bar.progress(i)
            # last_rows = new_rows
            time.sleep(0.05)

        progress_bar.empty()

        st.write(
            'A convolutional neural network, or CNN for short, is a type of neural network developed for working with '
            'two-dimensional image data.The CNN can be very effective at automatically extracting and learning '
            'features from one-dimensional sequence data such as univariate time series data. A CNN model can be used '
            'in a hybrid model with an LSTM backend where the CNN is used to interpret subsequences of input that '
            'together are provided as a sequence to an LSTM model to interpret.')

        st.write('The LSTM model has predicted that the next value in the sequence will be ' + str(yhat[0][0]))
        last_price = jse_value.iloc[-1]
        st.write('The current jamaica stock exchange index value is ' + str(last_price))

        difference = yhat[0][0] - jse_value.iloc[-1]

        st.write('This reflects a difference of ' + str(round(difference, 2)))

    if choice == 'Conv LSTM':
        url = 'https://www.jamstockex.com/market-data/download-data/index-history/main-market/JSE-Index/2010-08-08/2020-08-10'
        df = pd.read_html(url)

        jse = df[0]

        # New DF
        jse_value = jse['Value']
        raw_seq = jse_value

        def split_sequence(sequence, n_steps):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the sequence
                if end_ix > len(sequence) - 1:
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        # choose a number of time steps
        n_steps = 4
        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
        n_features = 1
        n_seq = 2
        n_steps = 2
        X = X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))

        # define model
        model = Sequential()
        model.add(
            ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        model.fit(X, y, epochs=500, verbose=0)

        x_input = array(jse_value.tail(4).tolist())
        x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = np.random.randn(1, 1)
        chart = st.line_chart(jse_value, use_container_width=True)

        for i in range(1, 101):
            new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)
            # chart.add_rows(new_rows)
            progress_bar.progress(i)
            # last_rows = new_rows
            time.sleep(0.05)

        progress_bar.empty()

        st.write(
            'ConvLSTM is related to the CNN-LSTM, where the convolutional reading of input is built directly into '
            'each LSTM unit. The ConvLSTM was developed for reading two-dimensional spatial-temporal data, '
            'but can be adapted for use with univariate time series forecasting.')

        st.write('The LSTM model has predicted that the next value in the sequence will be ' + str(yhat[0][0]))
        last_price = jse_value.iloc[-1]
        st.write('The current jamaica stock exchange index value is ' + str(last_price))

        difference = yhat[0][0] - jse_value.iloc[-1]

        st.write('This reflects a difference of ' + str(round(difference, 2)))


if __name__ == '__main__':
    main()
