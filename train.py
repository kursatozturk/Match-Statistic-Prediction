#%%
from new_utils import UTILITY
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, ConvLSTM2D, Conv2D, Reshape, Flatten, Dense, concatenate, MaxPool2D, Dropout, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K

def get_model(learning_rate=1e-5):
    # ht input
    input_layer_ht = Input(shape=(1, 2, 33), name='half_time')
    first_conv_b = Conv2D(filters=100, kernel_size=(1, 2), activation='relu')(input_layer_ht)
    first_conv_c = Conv2D(filters=100, kernel_size=(1, 3), padding='same', activation='relu')(input_layer_ht)
    first_layer_concat = concatenate([first_conv_b, first_conv_c], axis=3)
    batch_normalization_1 = BatchNormalization()(first_layer_concat)
    third_conv_a = Conv2D(filters=50, kernel_size=(1, 3), strides=(1, 3), activation='relu')(batch_normalization_1)
    

    # ft input
    input_layer_ft = Input(shape=(1, 2, 33), name='full_time')
    ft_first_conv_b = Conv2D(filters=100, kernel_size=(1, 2), activation='relu')(input_layer_ft)
    ft_first_conv_c = Conv2D(filters=100, kernel_size=(1, 3), padding='same', activation='relu')(input_layer_ft)
    ft_first_layer_concat = concatenate([ft_first_conv_b, ft_first_conv_c], axis=3)
    ft_batch_normalization_1 = BatchNormalization()(ft_first_layer_concat)
    ft_third_conv_a = Conv2D(filters=50, kernel_size=(1, 3), strides=(1, 3), activation='relu')(ft_batch_normalization_1)

    # ht output
    ht_conv_last = Conv2D(filters=50, kernel_size=(2, 3), activation='relu')(third_conv_a)
    ht_pool = MaxPool2D(pool_size=(1, 3))(ht_conv_last)
    ht_flat = Flatten()(ht_pool)
    ht_top_dense_1 = Dense(100, activation='relu')(ht_flat)
    ht_dropout = Dropout(0.5)(ht_top_dense_1)
    ht_top_dense_2 = Dense(50)(ht_dropout)
    ht_output = Dense(2, activation='relu', name='half_time_o')(ht_top_dense_2)


    # ft output
    ft_conv_last = Conv2D(filters=50, kernel_size=(2, 3), activation='relu')(ft_third_conv_a)    
    ft_pool = MaxPool2D(pool_size=(1, 3))(ft_conv_last)
    ft_flat = Flatten()(ft_pool)
    ft_top_dense_1 = Dense(100, activation='relu')(ft_flat)
    ft_dropout = Dropout(0.5)(ft_top_dense_1)
    ft_top_dense_2 = Dense(50, activation='relu')(ft_dropout)
    ft_output_relation_ht = concatenate([ft_top_dense_2, ht_output])
    ft_output = Dense(2, activation='relu', name='full_time_o')(ft_output_relation_ht)


    model = Model(inputs=(input_layer_ht, input_layer_ft), outputs=(ht_output, ft_output))
    opt = Adam(lr=learning_rate)
    losses = {
            'half_time_o': 'logcosh',
            'full_time_o': 'logcosh'
    }

    model.compile(opt, loss=losses)
    return model

def stat_model(learning_rate=1e-5):
    # first half stats input 
    first_half_input_stats = Input(shape=(2, 2, 33), name='first_half')
    first_half_cl_1 = Conv2D(filters=200, kernel_size=(1, 2), activation='relu')(first_half_input_stats)
    first_half_cl_2 = Conv2D(filters=200, kernel_size=(1, 3), activation='relu')(first_half_cl_1)
    first_half_bn_1 = BatchNormalization()(first_half_cl_2)
    first_half_cl_3_a = Conv2D(filters=200, kernel_size=(2, 3), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3_b = Conv2D(filters=200, kernel_size=(1, 5), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3 = concatenate([first_half_cl_3_a, first_half_cl_3_b], axis=2)

    # second half stats input
    second_half_input_stats = Input(shape=(2, 2, 33), name='second_half')
    second_half_cl_1 = Conv2D(filters=200, kernel_size=(1, 2), activation='relu')(second_half_input_stats)
    second_half_cl_2 = Conv2D(filters=200, kernel_size=(1, 3), activation='relu')(second_half_cl_1)
    second_half_bn_1 = BatchNormalization()(second_half_cl_2)
    second_half_cl_3_a = Conv2D(filters=200, kernel_size=(2, 3), padding='same', activation='relu')(second_half_bn_1)
    second_half_cl_3_b = Conv2D(filters=200, kernel_size=(1, 5), padding='same', activation='relu')(second_half_bn_1)
    second_half_cl_3 = concatenate([second_half_cl_3_a, second_half_cl_3_b], axis=2)

    pack = concatenate([first_half_cl_3, second_half_cl_3])
    pack_cl_1 = Conv2D(filters=100, kernel_size=(2, 4), padding='same', activation='relu')(pack)
    pack_bn_1 = BatchNormalization()(pack_cl_1)
    

    # first half stats output
    first_half_cl_4 = Conv2D(filters=50, kernel_size=(2, 3), padding='same', activation='relu')(pack_bn_1)
    first_half_mp_4 = MaxPool2D(pool_size=(1, 2))(first_half_cl_4)
    first_half_flat = Flatten()(first_half_mp_4)
    first_half_dense = Dense(100, activation='relu')(first_half_flat)
    first_half_dl = Dropout(0.5)(first_half_dense)
    first_half_output = Dense(66, activation='relu', name='first_half_o')(first_half_dl)


    # second half stats output
    second_half_cl_4 = Conv2D(filters=50, kernel_size=(2, 3), padding='same', activation='relu')(pack_bn_1)
    second_half_mp_4 = MaxPool2D(pool_size=(1, 2))(second_half_cl_4)
    second_half_flat = Flatten()(second_half_mp_4)
    second_half_dense = Dense(100, activation='relu')(second_half_flat)
    second_half_dl = Dropout(0.5)(second_half_dense)
    second_half_output = Dense(66, activation='relu', name='second_half_o')(second_half_dl)

    model = Model(inputs=(first_half_input_stats, second_half_input_stats), outputs=(first_half_output, second_half_output))
    opt = Adam(learning_rate)
    losses = {
        'first_half_o': 'logcosh',
        'second_half_o': 'logcosh'
    }
    model.compile(opt, loss = losses)
    return model





def load_data():
    data, stats = UTILITY.prepare_data()

    data = np.array(data)
    scores = np.array(scores)
    
    ht_scores = scores[:, :2]
    ft_scores = scores[:, 2:]

    ft_data = data[:, :, :1]
    ht_data = data[:, :, 1:]
    print(ft_data.shape)
    print(ht_data.shape)
    ft_data = ft_data.reshape(-1, 2, 33, 1)
    ht_data = ht_data.reshape(-1, 2, 33, 1)

    return ht_data, ft_data, ht_scores, ft_scores

def hyperparameter_try(learning_rate, batch_size):
    model = get_model(learning_rate)
    trial_history = model.fit(
        x = {
            'half_time': ht_data[:500],
            'full_time': ft_data[:500]
        },
        y = {
            'half_time_o': ht_scores[:500],
            'full_time_o': ft_scores[:500]
        },
        batch_size=batch_size,
        epochs=10,
        verbose=2
    )
    print(f'{learning_rate, batch_size}')

def predict(model, input_data, hts, fts):
    result = model.predict(input_data)
    total = 0
    ht_correct = 0
    ft_correct = 0
    for ht_a, ft_a, ht_s, ft_s in zip(result[0], result[1], hts, fts):
        for ht, ft, htr, ftr in zip(ht_a, ft_a, ht_s, ft_s):
            ht_i = int(ht)
            if abs(ht_i - ht) >= 0.5:
                ht_i += 1
            ft_i = int(ft)
            if abs(ft_i - ft) >= 0.5:
                ft_i += 1
            if ht_i == htr:
                ht_correct += 1
            if ft_i == ftr:
                ft_correct += 1
            total += 1
            print(f'predicted: ht: {ht_i}({ht}), ft: {ft_i}({ft})')
            print(f'real score: ht: {htr}, ft: {ftr}')
        print('__________________')
    print(f'{ht_correct} of {total} is correct for first-half')
    print(f'{ft_correct} of {total} is correct for second-half')
        

def save_data():
    np.save('DATA', (ht_data, ft_data))
    np.save('SCORES', (ht_scores, ft_scores))


#%%
data, score = UTILITY.stats2scores()
data = np.array(data)
print(data.shape)
data = data.reshape(-1, 4, 33, 1)
score = np.array(score)
ht_data = data[:, 0:2]
ft_data = data[:, 2:]
ht_scores = score[:, 0:2]
ft_scores = score[:, 2:]
ft_scores -= ht_scores

print (data.shape)
print (ht_data.shape)
print (ft_data.shape)
print(ht_scores.shape)
print(score.shape)
#%%
batch_size = 2
learning_rate = 1e-3
filepath = "models/model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]
model = get_model(learning_rate)
model.summary()

history = model.fit(
    x = {
        'half_time': ht_data[:500],
        'full_time': ft_data[:500]
    },
    y = {
        'half_time_o': ht_scores[:500],
        'full_time_o': ft_scores[:500]
    },
    batch_size=batch_size * 2,
    epochs=50,
    verbose=2,
)

history = model.fit(
    x = {
        'half_time': ht_data[500:-1000],
        'full_time': ft_data[500:-1000],
    },
    y= {
        'half_time_o': ht_scores[500:-1000],
        'full_time_o': ft_scores[500:-1000],
    },
    batch_size=batch_size,
    epochs=200,
    verbose=2,
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks_list,
)
#%%
history = model.history
print(model.metrics_names)
print(model.evaluate(x={'half_time': ht_data[-1000:], 'full_time':ft_data[-1000:]}, y={'half_time_o': ht_scores[-1000:], 'full_time_o': ft_scores[-1000:]}))
print(model.predict(x={'half_time': ht_data[-5:], 'full_time': ft_data[-5:]}), sep='\n')
print(ht_scores[-5:], ft_scores[-5:], sep='\n')

plt.plot(history.history['val_half_time_o_loss'])
plt.plot(history.history['val_full_time_o_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val_half_time_loss', 'val_full_time_loss'], loc='upper left')
plt.savefig('loss.png', bbox_inches="tight")
plt.show()


#%%
ht_data, ft_data, ht_scores, ft_scores = load_data()
save_data()
#%%
ht_data, ft_data = np.load('DATA.npy')
ht_scores, ft_scores = np.load('SCORES.npy')
ft_scores -= ht_scores

#%%

model = load_model('models/model.h5')

#%%
pred = model.predict(x={'half_time': ht_data[-5:-3], 'full_time': ft_data[-5:-3]})

print(pred)
print(ht_scores[-5:-3])
print(ft_scores[-5:-3])
#%%
print(model.metrics_names)
hist = model.evaluate(x={'half_time': ht_data[-5:-4], 'full_time': ft_data[-5:-4]},
                      y={'half_time_o': ht_scores[-5:-4], 'full_time_o': ft_scores[-5:-4]})
print(hist)
#%%

lrs = [1e-1, 1e-2, 1e-3, 5e-3, 1e-4, 1e-5]
bss = [1, 2, 4, 8, 16]
for lr in lrs:
    for bs in bss:
        hyperparameter_try(lr, bs)

#%%
predict(model,{'half_time': ht_data[-50:], 'full_time': ft_data[-50:]}, ht_scores[-50:], ft_scores[-50:])

#%%
model.evaluate(x={'half_time': ht_data[-50:], 'full_time': ft_data[-50:]},
                y={'half_time_o': ht_scores[-50:], 'full_time_o': ft_scores[-50:]})

#%%
data, stats = UTILITY.prepare_data()

data = np.array(data)
stats = np.array(stats)

print(data.shape)
print(stats.shape)
#%%
np.save('STATS_DATA', data)
np.save('STATS_OUT', stats)

#%%
data = np.load('STATS_DATA.npy')
stats = np.load('STATS_OUT.npy')

ht_data = data[:, :, :2]
ft_data = data[:, :, 2:]
ht_stats = stats[:, :1]
ft_stats = stats[:, 1:]

print(ht_stats.shape)
print(ft_stats.shape)
ht_stats = ht_stats.reshape(-1, 66)
ft_stats = ft_stats.reshape(-1, 66)
#%%
model = stat_model(1e-3)
model.summary()

#%%
history = model.fit(
    x={
        'first_half': ht_data[:500],
        'second_half': ft_data[:500]
    },
    y={
        'first_half_o' : ht_stats[:500],
        'second_half_o': ft_stats[:500]
    },
    epochs=25,
    batch_size=4,
    verbose=2,
    shuffle=True,
)

#%%
r = model.evaluate(
    x={
        'first_half': ht_data[-1000:],
        'second_half': ft_data[-1000:]
    },
    y={
        'first_half_o' : ht_stats[-1000:],
        'second_half_o': ft_stats[-1000:]
    })
print(r)
#%%
history = model.history
plt.plot(history.history['first_half_o_loss'])
plt.plot(history.history['val_first_half_o_loss'])
plt.title('first half loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['half_time_loss','val_half_time_loss'], loc='upper left')
plt.savefig('dummy_loss.png', bbox_inches="tight")
plt.show()

plt.title('second half loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['second_half_o_loss'])
plt.plot(history.history['val_second_half_o_loss'])
plt.legend(['full_time_loss',  'val_full_time_loss'])
plt.savefig('dummy1_loss.png', bbox_inches="tight")
plt.show()

#%%
filepath = 'models/stats2stats_second_phase.h5'
K.set_value(model.optimizer.lr, 1e-7)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]
history = model.fit(
    x={
        'first_half': ht_data[500:-1000],
        'second_half': ft_data[500:-1000]
    },
    y={
        'first_half_o' : ht_stats[500:-1000],
        'second_half_o': ft_stats[500:-1000]
    },
    epochs=200,
    batch_size=4,
    verbose=2,
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks_list,
)

#%%
r = model.predict(
    x={
        'first_half': ht_data[-2:-1],
        'second_half': ft_data[-2:-1]
    }
)
print(ht_stats[-2])
print(ft_stats[-2])
print(r)


#%%
model = load_model('models/stats2stats.h5')



data, scores = UTILITY.prepare_data()

data = np.array(data)
scores = np.array(scores)

print(data.shape)
print(scores.shape)
np.save('stats2scores', data)
np.save('scores', scores)
#%%
data = np.load('stats2scores.npy')
scores = np.load('scores.npy')
#data = data.reshape(-1, 4, 33, 1)
scores = np.array(scores)
ht_data = data[:, :, :2]
ft_data = data[:, :, 2:]
ht_scores = scores[:, :2]
ft_scores = scores[:, 2:]
ft_scores -= ht_scores

#%%

big_boy = the_model(5e-3)
init_hist = big_boy.fit(
    x = {
        'first_half': ht_data[:1000],
        'second_half': ft_data[:1000]
    },
    y = {
        'first_half_o': ht_scores[:1000],
        'second_half_o': ft_scores[:1000]
    },
    epochs=50,
    verbose=2
)

#%%
aaa_stats, aaa_scores = UTILITY.stats2scores()
aaa_stats = np.array(aaa_stats)
aaa_scores = np.array(aaa_scores)
a_ht_stat = aaa_stats[:, :1]
a_ft_stat = aaa_stats[:, 1:]

a_ht_score = aaa_scores[:, :2]
a_ft_score = aaa_scores[:, 2:]
#%%
score_model = get_model(1e-4)
init_hist = score_model.fit(
        x = {
            'half_time': a_ht_stat[:1000],
            'full_time': a_ft_stat[:1000]
        },
        y = {
            'half_time_o': a_ht_score[:1000],
            'full_time_o': a_ft_score[:1000]
        },
        batch_size=8,
        epochs=50,
        verbose=2
    )

#%%

filepath = "models/score_predictor.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]

history = score_model.fit(
        x = {
            'half_time': a_ht_stat[1000:-1000],
            'full_time': a_ft_stat[1000:-1000]
        },
        y = {
            'half_time_o': a_ht_score[1000:-1000],
            'full_time_o': a_ft_score[1000:-1000]
        },
        batch_size=8,
        epochs=200,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks_list,
    )


#%%
plt.plot(history.history['half_time_o_loss'])
plt.plot(history.history['val_half_time_o_loss'])
plt.title('first half loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['half_time_loss','val_half_time_loss'], loc='upper left')
plt.savefig('score_loss_first_half.png', bbox_inches="tight")
plt.show()

plt.title('second half loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['full_time_o_loss'])
plt.plot(history.history['val_full_time_o_loss'])
plt.legend(['full_time_loss',  'val_full_time_loss'])
plt.savefig('scory_loss_second_half.png', bbox_inches="tight")
plt.show()


plt.title('init loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(init_hist.history['half_time_o_loss'])
plt.plot(init_hist.history['full_time_o_loss'])
plt.legend(['half_time_loss',  'full_time_loss'])
plt.savefig('scory_loss_init.png', bbox_inches="tight")
plt.show()



def stat_model_new(learning_rate=1e-5):
    # first half stats input 
    first_half_input_stats = Input(shape=(2, 2, 33), name='first_half')
    first_half_cl_1 = Conv2D(filters=100, kernel_size=(1, 4), activation='relu')(first_half_input_stats)
    first_half_bn_1 = BatchNormalization()(first_half_cl_1)
    first_half_cl_3_a = Conv2D(filters=100, kernel_size=(2, 3), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3_b = Conv2D(filters=100, kernel_size=(1, 5), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3 = concatenate([first_half_cl_3_a, first_half_cl_3_b], axis=2)

    # second half stats input
    second_half_input_stats = Input(shape=(2, 2, 33), name='second_half')
    second_half_cl_1 = Conv2D(filters=100, kernel_size=(1, 4), activation='relu')(second_half_input_stats)
    second_half_bn_1 = BatchNormalization()(second_half_cl_1)
    second_half_cl_3_a = Conv2D(filters=100, kernel_size=(2, 3), padding='same', activation='relu')(second_half_bn_1)
    second_half_cl_3_b = Conv2D(filters=100, kernel_size=(1, 5), padding='same', activation='relu')(second_half_bn_1)
    second_half_cl_3 = concatenate([second_half_cl_3_a, second_half_cl_3_b], axis=2)


    # first half stats output
    first_half_cl_4 = Conv2D(filters=50, kernel_size=(2, 3), padding='same', activation='relu')(first_half_cl_3)
    first_half_mp_4 = MaxPool2D(pool_size=(1, 2))(first_half_cl_4)
    first_half_flat = Flatten()(first_half_mp_4)
    first_half_dense = Dense(100, activation='relu')(first_half_flat)
    first_half_dl = Dropout(0.5)(first_half_dense)
    first_half_output = Dense(66, activation='relu', name='first_half_o')(first_half_dl)


    # second half stats output
    second_half_cl_4 = Conv2D(filters=50, kernel_size=(2, 3), padding='same', activation='relu')(second_half_cl_3)
    second_half_mp_4 = MaxPool2D(pool_size=(1, 2))(second_half_cl_4)
    second_half_flat = Flatten()(second_half_mp_4)
    second_half_dense = Dense(100, activation='relu')(second_half_flat)
    second_half_dl = Dropout(0.5)(second_half_dense)
    second_half_concat_with_first = concatenate([second_half_dl, first_half_output], axis=0)
    second_half_output = Dense(66, activation='relu', name='second_half_o')(second_half_concat_with_first)

    model = Model(inputs=(first_half_input_stats, second_half_input_stats), outputs=(first_half_output, second_half_output))
    opt = Adam(learning_rate)
    losses = {
        'first_half_o': 'logcosh',
        'second_half_o': 'logcosh'
    }
    model.compile(opt, loss = losses)
    return model







history = model.fit(
    x={
        'first_half': ht_data[:500],
        'second_half': ft_data[:500]
    },
    y={
        'first_half_o' : ht_stats[:500],
        'second_half_o': ft_stats[:500]
    },
    epochs=25,
    batch_size=4,
    verbose=2,
    shuffle=True,
)

filepath = 'models/new_stat_models.h5'
K.set_value(model.optimizer.lr, 1e-7)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]
history = model.fit(
    x={
        'first_half': ht_data[500:-1000],
        'second_half': ft_data[500:-1000]
    },
    y={
        'first_half_o' : ht_stats[500:-1000],
        'second_half_o': ft_stats[500:-1000]
    },
    epochs=200,
    batch_size=4,
    verbose=2,
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks_list,
)

#%%
model = load_model('models/stats2stats_second_phase.h5')
from keras.utils import plot_model
plot_model(model, to_file='model.png')































#%%
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, ConvLSTM2D, Conv2D, Reshape, Flatten, Dense, concatenate, MaxPool2D, Dropout, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K

# Code used to train the model
def stat_model(learning_rate=1e-5):
    # first half stats input 
    first_half_input_stats = Input(shape=(2, 2, 33), name='first_half')
    first_half_cl_1 = Conv2D(filters=200, kernel_size=(1, 2), activation='relu')(first_half_input_stats)
    first_half_cl_2 = Conv2D(filters=200, kernel_size=(1, 3), activation='relu')(first_half_cl_1)
    first_half_bn_1 = BatchNormalization()(first_half_cl_2)
    first_half_cl_3_a = Conv2D(filters=200, kernel_size=(2, 3), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3_b = Conv2D(filters=200, kernel_size=(1, 5), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3 = concatenate([first_half_cl_3_a, first_half_cl_3_b], axis=2)

    # second half stats input
    second_half_input_stats = Input(shape=(2, 2, 33), name='second_half')
    second_half_cl_1 = Conv2D(filters=200, kernel_size=(1, 2), activation='relu')(second_half_input_stats)
    second_half_cl_2 = Conv2D(filters=200, kernel_size=(1, 3), activation='relu')(second_half_cl_1)
    second_half_bn_1 = BatchNormalization()(second_half_cl_2)
    second_half_cl_3_a = Conv2D(filters=200, kernel_size=(2, 3), padding='same', activation='relu')(second_half_bn_1)
    second_half_cl_3_b = Conv2D(filters=200, kernel_size=(1, 5), padding='same', activation='relu')(second_half_bn_1)
    second_half_cl_3 = concatenate([second_half_cl_3_a, second_half_cl_3_b], axis=2)

    pack = concatenate([first_half_cl_3, second_half_cl_3])
    pack_cl_1 = Conv2D(filters=100, kernel_size=(2, 4), padding='same', activation='relu')(pack)
    pack_bn_1 = BatchNormalization()(pack_cl_1)
    

    # first half stats output
    first_half_cl_4 = Conv2D(filters=50, kernel_size=(2, 3), padding='same', activation='relu')(pack_bn_1)
    first_half_mp_4 = MaxPool2D(pool_size=(1, 2))(first_half_cl_4)
    first_half_flat = Flatten()(first_half_mp_4)
    first_half_dense = Dense(100, activation='relu')(first_half_flat)
    first_half_dl = Dropout(0.5)(first_half_dense)
    first_half_output = Dense(66, activation='relu', name='first_half_o')(first_half_dl)


    # second half stats output
    second_half_cl_4 = Conv2D(filters=50, kernel_size=(2, 3), padding='same', activation='relu')(pack_bn_1)
    second_half_mp_4 = MaxPool2D(pool_size=(1, 2))(second_half_cl_4)
    second_half_flat = Flatten()(second_half_mp_4)
    second_half_dense = Dense(100, activation='relu')(second_half_flat)
    second_half_dl = Dropout(0.5)(second_half_dense)
    second_half_output = Dense(66, activation='relu', name='second_half_o')(second_half_dl)

    model = Model(inputs=(first_half_input_stats, second_half_input_stats), outputs=(first_half_output, second_half_output))
    opt = Adam(learning_rate)
    losses = {
        'first_half_o': 'logcosh',
        'second_half_o': 'logcosh'
    }
    model.compile(opt, loss = losses)
    return model

model = stat_model(1e-3)


# initialize model weights with overfitting a small data
history = model.fit(
    x={
        'first_half': ht_data[:500],
        'second_half': ft_data[:500]
    },
    y={
        'first_half_o' : ht_stats[:500],
        'second_half_o': ft_stats[:500]
    },
    epochs=25,
    batch_size=4,
    verbose=2,
    shuffle=True,
)

# do training 
filepath = 'models/new_stat_models.h5' # it was stats2stats.h5
#K.set_value(model.optimizer.lr, 1e-7) ## When loss become stable, decrease learning rate and start again
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]
history = model.fit(
    x={
        'first_half': ht_data[500:-1000],
        'second_half': ft_data[500:-1000]
    },
    y={
        'first_half_o' : ht_stats[500:-1000],
        'second_half_o': ft_stats[500:-1000]
    },
    epochs=200,
    batch_size=4,
    verbose=2,
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks_list,
)

# plot loss curves
plt.plot(history.history['first_half_o_loss'])
plt.plot(history.history['val_first_half_o_loss'])
plt.title('first half loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['half_time_loss','val_half_time_loss'], loc='upper left')
plt.savefig('dummy_loss.png', bbox_inches="tight")
plt.show()

plt.title('second half loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['second_half_o_loss'])
plt.plot(history.history['val_second_half_o_loss'])
plt.legend(['full_time_loss',  'val_full_time_loss'])
plt.savefig('dummy1_loss.png', bbox_inches="tight")
plt.show()









#%%

### Init necessary things
import numpy as np
from keras.models import load_model
K.set_image_data_format('channels_first')
def select_input(beg, end):
    return ht_data[beg:end], ft_data[beg:end], ht_stats[beg:end], ft_stats[beg:end]

def predict_stats(model, ht_input, ft_input, ht_output, ft_output):
    ht_prediction, ft_prediction = model.predict(
        x = {
            'first_half': ht_input,
            'second_half': ft_input
        }
    )

    print('Predicted Val(Real Val):\n')
    print('_________________________')
    for htpa, ftpa, htoa, ftoa in zip(ht_prediction, ft_prediction, ht_output, ft_output):
        for htp, ftp, hto, fto in zip(htpa, ftpa, htoa, ftoa):
            print(f'ht => {int(htp)}({hto}) - ft => {int(ftp)}({fto})')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Predicted Statistics Finished for one match.')



data = np.load('STATS_DATA.npy')
stats = np.load('STATS_OUT.npy')
training_ht_data = data[:-1000, :, :2]
training_ft_data = data[:-1000, :, 2:]
training_ht_stats = stats[:-1000, :1]
training_ft_stats = stats[:-1000, 1:]
ht_data = data[-1000:, :, :2]
ft_data = data[-1000:, :, 2:]
ht_stats = stats[-1000:, :1]
ft_stats = stats[-1000:, 1:]
ht_stats = ht_stats.reshape(-1, 66)
ft_stats = ft_stats.reshape(-1, 66)

model = load_model('models/stats2stats.h5')

#%%
# See samples

print('\tSample input looks like :\n')
print(' For first half:') 
(home_ht_avg_stats, home_ht_opp_avg_stats), (away_ht_avg_stats, away_ht_opp_avg_stats) = training_ht_data[0]
print(f' Home team\'s average statistics in previously played matches(first half only):\n {home_ht_avg_stats}\n')
print(f' Home team\'s opponents average statistics in previously played matches(first half only):\n {home_ht_opp_avg_stats}\n')
print(f' Away team\'s average statistics in previously played matches(first half only):\n {away_ht_avg_stats}\n')
print(f' Away team\'s opponents average statistics in previously played matches(first half only):\n {away_ht_opp_avg_stats}\n')

print(f' For second half:')
(home_ft_avg_stats, home_ft_opp_avg_stats), (away_ft_avg_stats, away_ft_opp_avg_stats) = training_ft_data[0]
print(f' Home team\'s average statistics in previously played matches(second half only):\n {home_ft_avg_stats}\n')
print(f' Home team\'s opponents average statistics in previously played matches(second half only):\n {home_ft_opp_avg_stats}\n')
print(f' Away team\'s average statistics in previously played matches(second half only):\n {away_ft_avg_stats}\n')
print(f' Away team\'s opponents average statistics in previously played matches(second half only):\n {away_ft_opp_avg_stats}\n')

print('Sample output looks like :')
print(f' For first half: {training_ht_stats[0]}')
print(f' For second half: {training_ft_stats[0]}')



#%%

# Evaluation of model

eval_result = model.evaluate(
    x = {
        'first_half': ht_data,
        'second_half': ft_data
    },
    y = {
        'first_half_o': ht_stats,
        'second_half_o': ft_stats
    }
)

for name, eval_val in zip(model.metrics_names, eval_result):
    print(f'{name}: {eval_val}')

#%%
# See some results model produce
beg = 0
end = 5
predict_stats(model, *select_input(beg, end))

#%%
def fully_connected():
    input_layer = Input(shape=(2, 4, 33))
    flat = Flatten()(input_layer)
    fc_l_1 = Dense(2000, activation='relu')(flat)
    fc_l_2 = Dense(1000, activation='relu')(fc_l_1)
    fc_l_3 = Dense(500, activation='relu')(fc_l_2)
    fc_l_4 = Dense(250, activation='relu')(fc_l_3)
    fc_l_5 = Dense(100, activation='relu')(fc_l_4)
    output = Dense(132, activation='relu')(fc_l_5)

    model  = Model(input_layer, output)

    loss = 'logcosh'
    opt = Adam()
    model.compile(opt, loss)
    return model

model = fully_connected()
model.summary()
fcout = stats.reshape(-1, 132)

history = model.fit(
    x=data[:500],
    y=fcout[:500],
    batch_size=4,
    epochs=25,
    verbose=2
)

#%%
history = model.fit(
    x=data[500:],
    y=fcout[500:],
    batch_size=4,
    epochs=20,
    verbose=2,
    validation_split=.2,
    shuffle=True
)
#%%
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(history.history['loss'])
plt.legend(['loss'])
#plt.savefig('scory_loss_second_half.png', bbox_inches="tight")
plt.show()

#%%
r = model.evaluate(
    x=data[-1000:],
    y=fcout[-1000:]
)
print(r)

#%%
pr = model.predict(x=data[-1:])
print(pr)
print(fcout[-1])

#%%
pr = pr[0]
print('Predicted(Real)')
for p, r in zip(pr, fcout[-1]):
    print(f'{int(p)}({r})')

#%%
def different_conv():
    input_layer = Input(shape=(2, 4, 33))
    conv1 = Conv2D(filters=500, kernel_size=(1, 2))(input_layer)
    conv2 = Conv2D(filters=500, kernel_size=(1, 4), activation='relu')(conv1)
    conv3 = Conv2D(filters=500, kernel_size=(2, 4), activation='relu', strides=(2, 1))(conv2)
    bn_l = BatchNormalization()(conv3)
    conv4 = Conv2D(filters=500, kernel_size=(1, 3), activation='relu')(bn_l)
    maxpool4 = MaxPool2D(pool_size=(1, 3))(conv4)
    conv5 = Conv2D(filters=500, kernel_size=(1, 2), activation='relu')(maxpool4)
    bn_l_2 = BatchNormalization()(conv5)
    conv6 = Conv2D(filters=500, kernel_size=(1, 4), activation='relu')(bn_l_2)
    maxpool6 = MaxPool2D(pool_size=(1, 4))(conv6)
    flat = Flatten()(maxpool6)
    fc_1 = Dense(500, activation='relu')(flat)
    drl = Dropout(0.5)(fc_1)
    output = Dense(132, activation='relu')(drl)

    model = Model(input_layer, output)

    loss = 'logcosh'
    opt = Adam()
    model.compile(opt, loss)
    return model

model = different_conv()
model.summary()

history = model.fit(
    x=data[:500],
    y=fcout[:500],
    batch_size=4,
    epochs=25,
    verbose=2
)

#%%
history = model.fit(
    x=data[500:],
    y=fcout[500:],
    batch_size=8,
    epochs=20,
    verbose=2,
    validation_split=.2,
    shuffle=True
)


#%%
from keras.layers import TimeDistributed
def for_half():

    first_half_input_stats = Input(shape=(2, 2, 33), name='first_half')
    first_half_cl_1 = Conv2D(filters=400, kernel_size=(1, 2), activation='relu')(first_half_input_stats)
    first_half_cl_2 = Conv2D(filters=400, kernel_size=(1, 3), activation='relu')(first_half_cl_1)
    first_half_bn_1 = BatchNormalization()(first_half_cl_2)
    first_half_cl_3_a = Conv2D(filters=400, kernel_size=(2, 3), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3_b = Conv2D(filters=400, kernel_size=(1, 5), padding='same', activation='relu')(first_half_bn_1)
    first_half_cl_3 = concatenate([first_half_cl_3_a, first_half_cl_3_b], axis=2)


    # first half stats output
    first_half_cl_4 = Conv2D(filters=200, kernel_size=(2, 3), padding='same', activation='relu')(first_half_cl_3)
    first_half_mp_4 = MaxPool2D(pool_size=(1, 2))(first_half_cl_4)
    first_half_flat = Flatten()(first_half_mp_4)
    first_half_dense = Dense(100, activation='relu')(first_half_flat)
    first_half_dl = Dropout(0.5)(first_half_dense)
    first_half_output = TimeDistributed(Dense(66, activation='relu', name='first_half_o')(first_half_dl)

    model = Model(first_half_input_stats, first_half_output)

    loss = 'logcosh'
    opt = Adam()
    model.compile(opt, loss)
    return model

model = for_half()
#%%
half_data = data[:, :, :2] + data[:, :, 2:]
halfout = stats[:, :1] + stats[:, 1:]
halfout = halfout.reshape(-1, 66)
#%%
history = model.fit(
    x=half_data[:500],
    y=halfout[:500],
    batch_size=4,
    epochs=25,
    verbose=2
)

#%%
history = model.fit(
    x=half_data[500:],
    y=halfout[500:],
    batch_size=8,
    epochs=20,
    verbose=2,
    validation_split=.2,
    shuffle=True
)

#%%



N, *_ = data.shape
half_data = np.zeros((2 * N, 2, 2, 33))
halfout = np.zeros((2 * N, 2, 33))
ht_data = data[:, :, :2]
ft_data = data[:, :, 2:]
ht_stats = stats[:, :1]
ft_stats = stats[:, 1:]

for i in range(0, 2 * N, 2):
    try:
        half_data[i] = ht_data[i//2]
        halfout[i] = ht_stats[i//2]
        half_data[i + 1] = ft_data[i//2]
        halfout[i + 1] = ft_stats[i//2]
    except Exception as e:
        print(e)
        print(i)
        break
    
    
halfout = halfout.reshape(-1, 66)
#%%
from keras.layers import ConvLSTM2D, TimeDistributed, Activation
K.set_image_data_format('channels_first')
def for_half_lstm(lr):

    layer_input_stats = Input(shape=(2, 2, 2, 33), name='PrevStats')
    
    layer_cl_1 = TimeDistributed(Conv2D(filters=200, kernel_size=(1, 5), activation='relu'))(layer_input_stats)
    layer_cl_2 = TimeDistributed(Conv2D(filters=200, kernel_size=(1, 4), activation='relu'))(layer_cl_1)
    layer_cl_3 = TimeDistributed(Conv2D(filters=200, kernel_size=(1, 3), activation='relu'))(layer_cl_2)
    layer_bn_1 = TimeDistributed(BatchNormalization())(layer_cl_2)
    layer_cl_4_a = TimeDistributed(Conv2D(filters=150, kernel_size=(2, 5), activation='relu'))(layer_bn_1)
    layer_cl_4_b = TimeDistributed(Conv2D(filters=150, kernel_size=(1, 5), activation='relu'))(layer_bn_1)
    layer_cl_4 = concatenate([layer_cl_4_a, layer_cl_4_b], axis=3)
    layer_cl_5 = TimeDistributed(Conv2D(filters=100, kernel_size=(2, 4), padding='same', activation='relu'))(layer_cl_4)
    layer_mp_4 = TimeDistributed(MaxPool2D(pool_size=(1, 2)))(layer_cl_5)
    first_half = ConvLSTM2D(filters=100, kernel_size=(2, 8), padding='same', activation='relu', return_sequences=True)(layer_mp_4)
    layer_flat = TimeDistributed(Flatten())(first_half)
    layer_dense = TimeDistributed(Dense(200, activation='relu'))(layer_flat)
    layer_dl = TimeDistributed(Dropout(0.5))(layer_dense)
    layer_output = TimeDistributed(Dense(66, name='statistics'))(layer_dl)
    output = Activation('relu')(layer_output)
    model = Model(layer_input_stats, output)

    loss = 'logcosh'
    opt = Adam(lr=lr)
    model.compile(opt, loss)
    return model






class KerasBatchGenerator(object):

    def __init__(self, data, out, num_steps, batch_size, skip_step=2):
        self.data = data
        self.out = out
        self.num_steps = num_steps
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, *self.data.shape[1:]))
        y = np.zeros((self.batch_size, self.num_steps, self.out.shape[-1]))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                y[i, :] = self.out[self.current_idx:self.current_idx + self.num_steps]
                self.current_idx += self.skip_step
            yield x, y

#%%
model = for_half_lstm(4e-4)
model.summary()
generator = KerasBatchGenerator(half_data[:2000], halfout[:2000], 2, 8, 2)
init_hist = model.fit_generator(
    generator.generate(),
    epochs=20,
    verbose=2,
    steps_per_epoch=250
    )
#%%
steps_per_epoch = (len(half_data) - 12000) // 16
validation_steps = 6000//16
train_generator = KerasBatchGenerator(half_data[2000:-10000], halfout[2000:-10000], 2, 4, 2)
valid_generator = KerasBatchGenerator(half_data[-10000:-4000], halfout[-10000:-4000], 2, 8, 2)

history = model.fit_generator(
    train_generator.generate(),
    validation_data=valid_generator.generate(),
    epochs=100,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)





#%%
test_generator = KerasBatchGenerator(half_data[-4000:], halfout[-4000:], 2, 8, 2)
results = model.evaluate_generator(test_generator.generate(), steps=4000//16)
print(results)
#%%

pred = model.predict(
    x=np.array([half_data[-2:]]),
)
real = halfout[-2:]
pred = pred[0]
for p, r in zip(pred, real):
    print('Pred(Real)')
    print('________________')
    for x,y in zip(p, r):
        print(f'{int(x)} => {y}')

#%%
loss_scores = """
    # Fully Connected: val_loss: 4.3510
    # LSTM 4.4181
    # Convolutional with one input/output: 4.8248
    # Convolutional with one input/output(only one half per time) 9.19
    # Convolutional with two input/output 4.2520

"""

#%%

#%%