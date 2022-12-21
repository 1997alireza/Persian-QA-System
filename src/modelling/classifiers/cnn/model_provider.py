from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten, Dropout, Concatenate, Activation
from keras.models import Model, Sequential
from keras.optimizers import RMSprop


def get_model(embedding_matrix, vocab_size, embedding_dim, len_labels_index, max_sequence_len):
    """
    One of the models defined in Convolutional Neural Networks for Sentence Classification paper by Yoon Kim
    """
    (kernel_size_list, number_of_filters, pool_size_list,
     dropout_list, optimizer) \
        = _param_selector()

    embedding_layer = Embedding(vocab_size + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_len,
                                trainable=False)

    print('Defining CNN model')

    input_node = Input(shape=(max_sequence_len, embedding_dim))
    conv_list = []
    for index, kernel_size in enumerate(kernel_size_list):
        filters = number_of_filters[index]
        pool_size = pool_size_list[index]
        conv = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input_node)
        pool = MaxPooling1D(pool_size=pool_size)(conv)
        flatten = Flatten()(pool)
        conv_list.append(flatten)

    if len(kernel_size_list) > 1:
        out = Concatenate()(conv_list)
    else:
        out = conv_list[0]

    graph = Model(inputs=input_node, outputs=out)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(dropout_list[0], input_shape=(max_sequence_len, embedding_dim)))
    model.add(graph)
    model.add(Dense(150))
    model.add(Dropout(dropout_list[1]))
    model.add(Activation('relu'))
    model.add(Dense(len_labels_index, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    return model


def _param_selector():
    kernel_size_list = [3, 4, 5]
    number_of_filters = [150, 150, 150]
    pool_size_list = [2, 2, 2]
    dropout_list = [0.25, 0.5]
    optimizer = RMSprop(lr=0.001, decay=0.0,
                        clipvalue=5.)
    return (kernel_size_list, number_of_filters, pool_size_list,
            dropout_list, optimizer)
