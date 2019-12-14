
import numpy as np
import sqlite3
import pickle

def team_events():
    conn = sqlite3.connect('all_statistics.db')
    cur1 = conn.cursor()
    cur2 = conn.cursor()
    teams = list(cur1.execute('select * from Team;'))
    team_event_list = {}
    for team_id, team_info in teams:
        event_list = list()
        home_team_events = list(cur2.execute('select * from Event where home_team_id=? order by Date', (team_id,)))
        away_team_events = list(cur2.execute('select * from Event where away_team_id=? order by Date', (team_id, )))
        i = 0
        j = 0
        l1 = len(home_team_events)
        l2 = len(away_team_events)
        while i < l1 and j < l2:
            if home_team_events[i][3] < away_team_events[j][3]:
                p = parse_stats(home_team_events[i][4], 'home')
                q = parse_stats(home_team_events[i][4], 'away')
                if p is None or q is None:
                    i += 1
                    continue
                event_list.append((p, q, home_team_events[i][3]))
                i += 1
            else:
                p = parse_stats(away_team_events[j][4], 'away')
                q = parse_stats(away_team_events[j][4], 'home')
                if p is None or q is None:
                    j += 1
                    continue
                event_list.append((p, q, away_team_events[j][3]))
                j += 1
        while i < l1:
            p = parse_stats(home_team_events[i][4], 'home')
            q = parse_stats(home_team_events[i][4], 'away')
            if p is None or q is None:
                i += 1
                continue
            event_list.append((p, q, home_team_events[i][3]))
            i += 1
        while j < l2:
            p = parse_stats(away_team_events[j][4], 'away')
            q = parse_stats(away_team_events[j][4], 'home')
            if p is None or q is None:
                j += 1
                continue
            event_list.append((p, q, away_team_events[j][3]))
            j += 1
        team_event_list[team_id] = event_list
    conn.close()
    return team_event_list

def parse_stats(event_dict, home_away):
    keys = ['awayPasses', 'homePasses', 'awayDuelWon', 'homeDuelWon', 
    'awayDuelLost', 'awayOffsides', 'awayThrowIns', 'homeDuelLost', 
    'homeOffsides', 'homeThrowIns', 'awayAerialWon', 'awayFreeKicks', 
    'awayGoalKicks', 'homeAerialWon', 'homeFreeKicks', 'homeGoalKicks', 
    'awayAerialLost', 'awayFastBreaks', 'homeAerialLost', 'homeFastBreaks', 
    'awayCornerKicks', 'awayHitWoodwork', 'awayShotsOnGoal', 'homeCornerKicks', 
    'homeHitWoodwork', 'homeShotsOnGoal', 'awayAttInBoxGoal', 'awayAttInBoxPost', 
    'awayShotsOffGoal', 'homeAttInBoxGoal', 'homeAttInBoxPost', 'homeShotsOffGoal',
    'awayKeeperSweeper', 'homeKeeperSweeper', 'awayAccuratePasses', 'awayBallPossession',
    'awayDuelWonPercent', 'homeAccuratePasses', 'homeBallPossession', 'homeDuelWonPercent',
    'awayGoalkeeperSaves', 'homeGoalkeeperSaves', 
    'awayAerialWonPercent', 'awayAttInsideBoxMiss', 'awayAttOutBoxBlocked', 'awayBigChanceCreated',
    'awayTotalShotsOnGoal', 'homeAerialWonPercent', 'homeAttInsideBoxMiss', 'homeAttOutBoxBlocked', 
    'homeBigChanceCreated', 'homeTotalShotsOnGoal', 'awayAttOutsideBoxMiss', 'homeAttOutsideBoxMiss',
    'awayAttInsideBoxTarget', 'homeAttInsideBoxTarget', 'awayAttOutsideBoxTarget', 
    'awayTotalShotsInsideBox', 'homeAttOutsideBoxTarget', 'homeTotalShotsInsideBox', 
    'awayTotalShotsOutsideBox', 'homeTotalShotsOutsideBox', 'awayAccuratePassesPercent', 
    'awayBlockedScoringAttempt', 'homeAccuratePassesPercent', 'homeBlockedScoringAttempt']

    event_dict = pickle.loads(event_dict)
    stats = event_dict['stats']
    first_half = list()
    second_half = list()
    if stats == {}:
        return None
    for key in keys:
        if key.startswith(home_away):
            try:
                val_f = stats['period1'][key]
            except KeyError:
                val_f = 0
            try:
                val_s = stats['period2'][key]
            except KeyError:
                val_s = 0
            first_half.append(val_f)
            second_half.append(val_s)
    return (first_half, second_half)
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

def stats2scores_parser(event_dict):
    keys = ['awayPasses', 'homePasses', 'awayDuelWon', 'homeDuelWon', 
    'awayDuelLost', 'awayOffsides', 'awayThrowIns', 'homeDuelLost', 
    'homeOffsides', 'homeThrowIns', 'awayAerialWon', 'awayFreeKicks', 
    'awayGoalKicks', 'homeAerialWon', 'homeFreeKicks', 'homeGoalKicks', 
    'awayAerialLost', 'awayFastBreaks', 'homeAerialLost', 'homeFastBreaks', 
    'awayCornerKicks', 'awayHitWoodwork', 'awayShotsOnGoal', 'homeCornerKicks', 
    'homeHitWoodwork', 'homeShotsOnGoal', 'awayAttInBoxGoal', 'awayAttInBoxPost', 
    'awayShotsOffGoal', 'homeAttInBoxGoal', 'homeAttInBoxPost', 'homeShotsOffGoal',
    'awayKeeperSweeper', 'homeKeeperSweeper', 'awayAccuratePasses', 'awayBallPossession',
    'awayDuelWonPercent', 'homeAccuratePasses', 'homeBallPossession', 'homeDuelWonPercent',
    'awayGoalkeeperSaves', 'homeGoalkeeperSaves', 
    'awayAerialWonPercent', 'awayAttInsideBoxMiss', 'awayAttOutBoxBlocked', 'awayBigChanceCreated',
    'awayTotalShotsOnGoal', 'homeAerialWonPercent', 'homeAttInsideBoxMiss', 'homeAttOutBoxBlocked', 
    'homeBigChanceCreated', 'homeTotalShotsOnGoal', 'awayAttOutsideBoxMiss', 'homeAttOutsideBoxMiss',
    'awayAttInsideBoxTarget', 'homeAttInsideBoxTarget', 'awayAttOutsideBoxTarget', 
    'awayTotalShotsInsideBox', 'homeAttOutsideBoxTarget', 'homeTotalShotsInsideBox', 
    'awayTotalShotsOutsideBox', 'homeTotalShotsOutsideBox', 'awayAccuratePassesPercent', 
    'awayBlockedScoringAttempt', 'homeAccuratePassesPercent', 'homeBlockedScoringAttempt']

    event_dict = pickle.loads(event_dict)
    stats = event_dict['stats']
    score = parse_score(event_dict['score'])
    if score is None:
        return None
    first_half = [list(), list()]
    second_half = [list(), list()]
    
    if stats == {}:
        return None
    for key in keys:
        try:
            val_f = stats['period1'][key]
        except KeyError:
            val_f = 0
        try:
            val_s = stats['period2'][key]
        except KeyError:
            val_s = 0
        if key.startswith('home'):
            first_half[0].append(val_f)
            second_half[0].append(val_s)
        else:
            first_half[1].append(val_f)
            second_half[1].append(val_s)   
        

    return (first_half, second_half), score
class UTILITY:
    team_event_list = None
    @staticmethod
    def parse_statistics_of_team(team_id, date):
        try:
            event_list = UTILITY.team_event_list[team_id]
        except KeyError:
            return None
        ht_stats_h = list()
        ft_stats_h = list()
        ht_stats_a = list()
        ft_stats_a = list()
        for (first_half_h, second_half_h),(first_half_a, second_half_a), e_date in event_list:
            if date <= e_date:
                break
            ht_stats_h.append(first_half_h)
            ft_stats_h.append(second_half_h)
            ht_stats_a.append(first_half_a)
            ft_stats_a.append(second_half_a)
        



        if ht_stats_h == [] or ht_stats_a == []:
            return None
        ht_stats_h = ht_stats_h[-5:]
        ft_stats_h = ft_stats_h[-5:]
        ht_stats_a = ht_stats_a[-5:]
        ft_stats_a = ft_stats_a[-5:]
        
        ht_stats_h = np.array(ht_stats_h).mean(axis=0)
        ft_stats_h = np.array(ft_stats_h).mean(axis=0)
        ht_stats_a = np.array(ht_stats_a).mean(axis=0)
        ft_stats_a = np.array(ft_stats_a).mean(axis=0)
        return list(ht_stats_h), list(ht_stats_a), list(ft_stats_h), list(ft_stats_a)

    @staticmethod
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

    @staticmethod
    def prepare_data():
        UTILITY.team_event_list = team_events()
        conn = sqlite3.connect('all_statistics.db')
        cur1 = conn.cursor()
        events = list(cur1.execute('select * from Event;'))
        data = []
        scores = []
        for e_id, h_id, a_id, date, event_info in events:
            home_stats = UTILITY.parse_statistics_of_team(h_id, date)
            away_stats = UTILITY.parse_statistics_of_team(a_id, date)
            if home_stats is None or away_stats is None:
                continue
            x = stats2scores_parser(event_info)
            if x is None:
                continue
            stat, score = x
            data.append([home_stats, away_stats])
            scores.append(score)
            #winners.append(winner)
        return data, scores

    @staticmethod
    def stats2scores():
        events_stats = []
        scores = []
        conn = sqlite3.connect('all_statistics.db')
        cur = conn.cursor()
        events = cur.execute('select * from Event;')
        for _, _, _, date, event_info  in events:
            x = stats2scores_parser(event_info)
            if x is None:
                continue
            stats, score = x
            events_stats.append(stats)
            scores.append(score)
        return events_stats, scores
