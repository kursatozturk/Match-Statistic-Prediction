import numpy as np
import sqlite3
import pickle
from keras.layers import Input, ConvLSTM2D, Conv2D, Flatten, Dense, concatenate, MaxPool2D, Dropout, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adam

def get_modeltt():
    
    input_layer = Input(shape=(2, 205, 1))
    first_conv_b = Conv2D(filters=200, kernel_size=(1, 20), strides=(1, 5), activation='relu')(input_layer)
    first_conv_c = Conv2D(filters=200, kernel_size=(2, 5), padding='same', activation='relu')(input_layer)
    first_layer_concat = concatenate([first_conv_b, first_conv_c], axis=2)
    batch_normalization_1 = BatchNormalization()(first_layer_concat)

    second_conv_a = Conv2D(filters=100, kernel_size=(1, 11),strides=(1, 11), activation='relu')(batch_normalization_1)
    second_conv_b = Conv2D(filters=100, kernel_size=(2, 4), padding='same', activation='relu')(batch_normalization_1)
    second_layer_concat = concatenate([second_conv_a, second_conv_b], axis=2)
    batch_normalization_2 = BatchNormalization()(second_layer_concat)

    third_conv_a = Conv2D(filters=100, kernel_size=(1, 3), strides=(1, 3), activation='relu')(batch_normalization_2)
    third_conv_b = Conv2D(filters=100, kernel_size=(2, 4), padding='same', activation='relu')(batch_normalization_2)
    third_conv_c = Conv2D(filters=100, kernel_size=(1, 5), activation='relu')(batch_normalization_2)
    third_layer_concat = concatenate([third_conv_a, third_conv_b, third_conv_c], axis=2)
    fourth_conv = Conv2D(filters=200, kernel_size=(1, 4), padding='same', activation='relu')(third_layer_concat)
    batch_normalization_3 = BatchNormalization()(fourth_conv)
    seventh_a_conv = Conv2D(filters=200, kernel_size=(1,3), strides=(1, 3), activation='relu')(batch_normalization_3)
    batch_normalization_4 = BatchNormalization()(seventh_a_conv)
    # score branch


    score_conv_0 = Conv2D(filters=100, kernel_size=(1, 5), activation='relu')(batch_normalization_4)
    #ht score
    score_conv_3 = Conv2D(filters=100, kernel_size=(1, 4), padding='same', activation='relu')(score_conv_0)
    score_conv_4 = Conv2D(filters=100, kernel_size=(1, 2), padding='same', activation='relu')(score_conv_3)
    score_conv_5 = Conv2D(filters=100, kernel_size=(1, 3), strides=(1, 3), activation='relu')(score_conv_4)
    batch_score_l = BatchNormalization()(score_conv_5)
    score_conv_6 = Conv2D(filters=100, kernel_size=(1, 4), strides=(1, 2), activation='relu')(batch_score_l)

    flat_h = Flatten()(score_conv_4)
    dense_h_1 = Dense(100, activation='relu')(flat_h)
    dropout_ht_1 = Dropout(0.5)(dense_h_1)
    output_0 = Dense(8, activation='softmax', name='ht_score')(dropout_ht_1)

    #ft score
    score_conv_1 = Conv2D(filters=200, kernel_size=(1, 4), padding='same', activation='relu')(score_conv_0)
    score_conv_2 = Conv2D(filters=200, kernel_size=(1, 2), strides=(1, 2), activation='relu')(score_conv_1)
    score_conv_1_a = Conv2D(filters=200, kernel_size=(2, 2), activation='relu')(score_conv_2)
    score_conv_1_b = Conv2D(filters=200, kernel_size=(1, 4), strides=(1, 2), activation='relu')(score_conv_1_a)
    score_conv_1_c = Conv2D(filters=200, kernel_size=(1, 3), strides=(1, 2), activation='relu')(score_conv_1_b)
    score_conv_1_d = Conv2D(filters=100, kernel_size=(1, 2), strides=(1, 2), activation='relu')(score_conv_1_c)


    flat = Flatten()(score_conv_1_d)
    dense_1 = Dense(100, activation='relu')(flat)
    dl = Dropout(0.5)(dense_1)
    dense_2 = Dense(60, activation='relu')(dl)
    dl_2 = Dropout(0.5)(dense_2)
    output_1 = Dense(8, activation='softmax', name='ft_score')(dl_2)

    # winner branch
    w_fourth_conv = Conv2D(filters=50, kernel_size=(1, 3), activation='relu')(batch_normalization_4)
    w_batch_normalization_2 = BatchNormalization()(w_fourth_conv)
    w_fourth_pooling = MaxPool2D(pool_size=(1, 3))(w_batch_normalization_2)
    w_fifth_conv = Conv2D(filters=100, kernel_size=(2, 4), padding='same', activation='relu')(w_fourth_pooling)
    flat_w = Flatten()(w_fifth_conv)

    # ht winner
    dense_w_1 = Dense(100, activation='relu')(flat_w)
    dl_w = Dropout(0.5)(dense_w_1)
    output_2 = Dense(3, activation='softmax', name='ht_winner')(dl_w)

    # ft winner 
    dense_w2_1 = Dense(100, activation='relu')(flat_w)
    dl_w_2 = Dropout(0.5)(dense_w2_1)
    output_3 = Dense(3, activation='softmax', name='ft_winner')(dl_w_2)


    model = Model(inputs=(input_layer,), outputs=(output_0, output_1, output_2, output_3 ))
    s = SGD(lr=3e-3, momentum=0.90, decay=0.99, nesterov=True)
    opt = Adam(lr=1e-4)
    losses = {
        'ht_winner' : 'categorical_crossentropy',
        'ft_winner' : 'categorical_crossentropy',
        'ht_score' : 'categorical_crossentropy',
        'ft_score' : 'categorical_crossentropy',
    }

    model.compile(opt, loss=losses, metrics=['acc'])
    return model
def eliminate_zeros(data, label):
    r_data = []
    r_label = []
    for (h, a), l in zip(data, label):
        ch = 0
        ca = 0
        for i in h:
            if i == 0:
                ch += 1
        for i in a:
            if i == 0:
                ca += 1
        if ch == 205 or ca == 205:
            continue
        else:
            r_data.append([h, a])
            r_label.append(l)
    return r_data, r_label
            

def vectorize(stat_dict):
    
    if 'msg' in stat_dict.keys():
        return None
    if stat_dict['minutesPlayed'] == 0:
        # player did not played in match, so there is no statistics affected score!!!
        return None
    arr = list()
    player_keys = ['goalAssist','goals', 'shotsBlocked', 'shotsOffTarget', 'shotsOnTarget', 
    'totalContest', 'challengeLost', 'interceptionWon', 
    'outfielderBlock', 'totalClearance', 'totalTackle', 
    'dispossessed', 'fouls', 'totalDuels', 'wasFouled', 'accuratePass', 
    'keyPass', 'totalCross', 'totalLongBalls', 'minutesPlayed']
    goal_keeper_keys = ['goodHighClaim', 'punches', 'runsOut', 'saves', 'minutesPlayed']
    keys = None
    if goal_keeper_keys[0] in stat_dict.keys():
        keys = goal_keeper_keys
    else:
        keys = player_keys
    for key in keys:
        val = stat_dict[key]
        if type(val) is list:
            arr.append(int(val[0]))
        elif type(val) is int or type(val) is str:
            arr.append(int(val))
        if key == 'minutesPlayed':
            arr[-1]/= 90.
    return arr

def player_dict():
    player_stats = dict()
    conn = sqlite3.connect('all_statistics.db')
    cur = conn.cursor()
    players = list(cur.execute('select * from Player;'))
    for player in players:
        pid = player[0]
        stats = cur.execute('select S.stats, E.date from Statistics S, Event E where S.event_id == E.event_id AND S.player_id = ? order by E.date;', (pid, ))
        try:
            stats = list(stats)
        except Exception as e:
            print(e)
            continue
        player_stats[pid] = []
        for stat, date in stats:
            stat = pickle.loads(stat)
            stat = vectorize(stat)
            if stat is None:
                continue
            else:
                player_stats[pid].append((stat, date))
    conn.close()
    return player_stats

def iy_ms(arr):
    results = []
    for hc1, ac1, hc2, ac2 in arr:
        result = [0, 0, 0, 0, 0, 0]
        if hc1 > ac1:
            result[0] = 1
        elif hc1 == ac1:
            result[1] = 1
        else:
            result[2] = 1
        if hc2 > ac2:
            result[3] = 1
        elif hc2==ac2:
            result[4] = 1
        else:
            result[5] = 1
        results.append(result)
    return results
def parse_score(score_dict):
    try:
        hc1 = score_dict['home_score']['period1']
        hc2 = score_dict['home_score']['normaltime']
        ac1 = score_dict['away_score']['period1']
        ac2 = score_dict['away_score']['normaltime']
        result = [hc1, ac1, hc2, ac2]
    except KeyError:
        return None
    return result
def determine_interval(hc1):
    if hc1 == 0 or hc1 == 1:
        return 0
    elif hc1 == 2 or hc1 == 3:
        return 1
    elif hc1 >= 4 and hc1 <= 6:
        return 2
    else:
        return 3
def goal_intervals(arr):
    results = []
    halftime = []
    fulltime = []
    for hc1, ac1, hc2, ac2 in arr:
        h_ht_result = [0, 0, 0, 0]
        h_ft_result = [0, 0, 0, 0]
        a_ht_result = [0, 0, 0, 0]
        a_ft_result = [0, 0, 0, 0]
        h_ht_result[determine_interval(hc1)] = 1
        h_ft_result[determine_interval(hc2)] = 1
        a_ht_result[determine_interval(ac1)] = 1
        a_ft_result[determine_interval(ac2)] = 1
        halftime.append(h_ht_result + a_ht_result)
        fulltime.append(h_ft_result + a_ft_result)
    return halftime, fulltime


def last_matches(stat_list, date):
    if stat_list is None:
        return None
    result = []
    for stat in stat_list:
        if date >= stat[1]:
            break
        result.append(stat[0])
    if result == []:
        return None
    result = np.array(result)
    result = result[-5:]
    result = np.mean(result, axis=0)
    return [list(result)]

def prepare_data():
    conn = sqlite3.connect('all_statistics.db')
    cur = conn.cursor()
    # first fetch events

    events = list(cur.execute('select * from Event;'))
    pl_dict = player_dict()
    event_stats = list()
    scores = list()
    idx = -1
    for event in events:
        idx += 1
        home_statistics_total = list()
        away_statistics_total = list()
        eid, hid, aid, date, event_info = event
        event_info = pickle.loads(event_info)

        if event_info.get('lineups') is None:
            print(f'{eid} event cannot be fetched: LACK OF LINEUPS')
            continue
        home, away = event_info.get('lineups')
        if len(home) < 11 or len(away) < 11:
            continue

        gkh = home[0]['id']
        gka = away[0]['id']
        try:
            phstat = last_matches(pl_dict[gkh], date)
            if phstat is None:
                phstat = [[0] * 5]
        except KeyError:
            phstat = [[0] * 5]
        try:
            pastat = last_matches(pl_dict[gka], date)
            if pastat is None:
                pastat = [[0] * 5]
        except KeyError:
            pastat = [[0] * 5]

        if len(pastat[0]) != 5:
            continue
        if len(phstat[0]) != 5:
            continue

        for ph in phstat:
            home_statistics_total += ph
        for pa in pastat:
            away_statistics_total += pa

        for  player_h, player_a in (zip(home[1:11], away[1:11])):
            phid = player_h['id']
            paid = player_a['id']
            try:
                phstat = last_matches(pl_dict[phid], date)
                if phstat is None:
                    phstat = [[0] * 20]
            except KeyError:
                phstat = [[0] * 20]
            try:
                pastat = last_matches(pl_dict[paid], date)
                if pastat is None:
                    pastat = [[0] * 20]
            except KeyError:
                pastat = [[0] * 20]
            except Exception as e:
                print(e)
                continue
            for ph in phstat:
                home_statistics_total += ph
            for pa in pastat:
                away_statistics_total += pa
        
        score = parse_score(event_info.get('score'))
        if score is not None:
            scores.append(score)
            event_stats.append((home_statistics_total, away_statistics_total))       
    return event_stats, scores

if __name__ == '__main__':

    event_stats, scores = prepare_data()
   
    event_stats, scores = eliminate_zeros(event_stats, scores)
    event_stats = np.array(event_stats)
    scores = np.array(scores)
    print(event_stats.shape)
    print(scores.shape)