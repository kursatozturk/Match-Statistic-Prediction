import sqlite3
from threading import Thread, RLock
import pickle
teams = list()
events = list()
players = list()
statistics = list()
data_lock = RLock()

def reader(num):
	global teams, events, players, statistics
	try:
		conn = sqlite3.connect(f'statistics_{num}.db')
		cur = conn.cursor()
		t = list(cur.execute('SELECT * FROM Team;'))
		p = list(cur.execute('SELECT * FROM Player;'))
		e = list(cur.execute('SELECT * FROM Event;'))
		s = list(cur.execute('SELECT * FROM Statistics;'))

		data_lock.acquire()
		teams += t
		events += e
		players += p
		statistics += s
		data_lock.release()
	except Exception as e:
		print(e)
	conn.close()

def create_db():
	conn = sqlite3.connect('all_statistics.db')
	cur = conn.cursor()
	cur.execute("""
	            CREATE TABLE Player(
	                player_id INTEGER PRIMARY KEY,
	                player_info BLOB
	            );"""
	            )
	cur.execute("""
	            CREATE TABLE Team(
	                team_id INTEGER PRIMARY KEY,
	                team_info BLOB
	            );"""
	            )
	cur.execute("""
	            CREATE TABLE Event(
	                event_id INTEGER PRIMARY KEY,
	                home_team_id INTEGER,
	                away_team_id INTEGER,
	                DATE date,
	                event_info BLOB,
	                FOREIGN KEY(home_team_id) REFERENCES Team,
	                FOREIGN KEY(away_team_id) REFERENCES Team
	            );"""
	            )
	cur.execute("""
	            CREATE TABLE Statistics(
	                player_id INTEGER,
	                event_id INTEGER,
	                stats BLOB,
	                FOREIGN KEY(player_id) REFERENCES Player,
	                FOREIGN KEY(event_id) REFERENCES Event,
	                PRIMARY KEY(player_id, event_id)
	            );"""
	        )
	conn.close()
if __name__ == '__main__':
	#create_db()
	threads = list()
	for i in range(3):
		t = Thread(target=reader, args=(i,))
		threads.append(t)
		t.start()
	for i in range(3):
		t.join()
	conn = sqlite3.connect('all_statistics.db')
	cur = conn.cursor()
	for s, e, p, t in zip(statistics, events, players, teams):
		try:
			cur.execute('INSERT INTO Statistics VALUES(?, ?, ?);', s)
		except sqlite3.IntegrityError:
			pass
		try:
			cur.execute('INSERT INTO Event Values(?, ?, ?, ?, ?);', e)
		except sqlite3.IntegrityError:
			pass
		try:
			cur.execute('INSERT INTO Player VALUES(?, ?);', p)
		except sqlite3.IntegrityError:
			pass
		try:
			cur.execute('INSERT INTO Team VALUES(?, ?);', t)
		except sqlite3.IntegrityError:
			pass
		except Exception as e:
			print(e)
	conn.commit()
	conn.close()

    
    
