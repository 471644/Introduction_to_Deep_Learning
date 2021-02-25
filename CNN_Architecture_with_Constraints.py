def cnn_model():
    model = Sequential()

 

    model.add(
        Convolution1D(128,
                      8,
                      activation='relu',
                      padding='valid',
                      input_shape=(93, 1)))
    model.add(
        Convolution1D(128,
                      8,
                      activation='relu',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))

 

    model.add(
        Convolution1D(64,
                      5,
                      activation='relu',
                      padding='valid',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01)))
    model.add(
        Convolution1D(64,
                      5,
                      activation='relu',
                      padding='valid',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))

 

    model.add(
        Convolution1D(32,
                      3,
                      activation='relu',
                      padding='valid',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(
        Convolution1D(32,
                      3,
                      activation='relu',
                      padding='valid',
                      kernel_regularizer=regularizers.l2(0.01),
                      bias_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling1D(pool_size=2))

 

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model