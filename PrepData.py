from multiprocessing import connection
import sqlite3
import json
from datetime import datetime
import pandas as pd
import os

rawDataPath = 'RawData/'
preparedDataPath = 'PreparedData/'

timeframe = '2018-10'
sqlTransaction = []

connection = sqlite3.connect(preparedDataPath + '{}.db'.format(timeframe))
c = connection.cursor()

def createTable():
    c.execute("""CREATE TABLE IF NOT EXISTS parentReply(parentId TEXT PRIMARY KEY,
                                                        commentId TEXT UNIQUE,
                                                        parent TEXT,
                                                        comment TEXT,
                                                        subreddit TEXT,
                                                        unix INT,
                                                        score INT)""")

def formatData(data):
    return data.replace("\n", " newLineChar ").replace("\r", " newLineChar ").replace('"', "'")

def findParent(pid):
    try:
        sql = "SELECT comment FROM parentReply WHERE commentId = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        # print("find parent", e)
        return False

def findExistingScore(pid):
    try:
        sql = "SELECT score FROM parentReply WHERE parentId = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result != None:
            return result[0]
        else: return False
    except Exception as e:
        # print("find parent", e)
        return False

def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]' or data == '[removed]':
        return False
    else:
        return True

def transactionBldr(sql):
    global sqlTransaction
    sqlTransaction.append(sql)
    if len(sqlTransaction) > 1000:
        c.execute("BEGIN TRANSACTION")
        for s in sqlTransaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sqlTransaction = []

def sqlInsertReplaceComment(commentId, parentId, parentData, body, subreddit, createdUtc, score):
    try:
        sql = """UPDATE parentReply SET parentId = {}, commentId = {}, parent = {}, comment = {}, unix = {}, score = {} WHERE parentId = {};""".format(parentId, commentId, parentData, body, subreddit, int(createdUtc), score, parentId)
        transactionBldr(sql)
    except Exception as e:
        print('s-UPDATE insertion', str(e))

def sqlInsertHasParent(commentId, parentId, parentData, body, subreddit, createdUtc, score):
    try:
        sql = """INSERT INTO parentReply (parentId, commentId, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentId, commentId, parentData, body, subreddit, int(createdUtc), score)
        transactionBldr(sql)
    except Exception as e:
        print('s-PARENT insertion', str(e))

def sqlInsertNoParent(commentId, parentId, body, subreddit, createdUtc, score):
    try:
        sql = """INSERT INTO parentReply (parentId, commentId, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentId, commentId, body, subreddit, int(createdUtc), score)
        transactionBldr(sql)
    except Exception as e:
        print('s-NOPARENT insertion', str(e))


createTable()
rowCounter = 0
pairedRows = 0
with open(rawDataPath + '/RC_{}'.format(timeframe), buffering=1000) as f:
    for row in f:
        rowCounter += 1
        row = json.loads(row)
        parentId = row['parent_id']
        commentId = row['link_id']
        body = formatData(row['body'])
        createdUtc = row['created_utc']
        score = row['score']
        subreddit = row['subreddit']

        parentData = findParent(parentId)

        if score >= 2:
            if acceptable(body):
                existingCommentScore = findExistingScore(parentId)
                if existingCommentScore:
                    if score > existingCommentScore:
                        sqlInsertReplaceComment(commentId, parentId, parentData, body, subreddit, createdUtc, score)
                else:
                    if parentData:
                        sqlInsertHasParent(commentId, parentId, parentData, body, subreddit, createdUtc, score)
                        pairedRows += 1
                    else:
                        sqlInsertNoParent(commentId, parentId, body, subreddit, createdUtc, score)
        if rowCounter % 10000 == 0:
            print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(rowCounter, pairedRows, str(datetime.now())))


timeframes = ['2018-10']
for timeframe in timeframes:
    if not os.path.exists('PreparedData/{}.db'.format(timeframe)):
        connection = sqlite3.connect('PreparedData/{}.db'.format(timeframe))
        c = connection.cursor()
        limit = 5000
        lastUnix = 0
        curLength = limit
        counter = 0
        testDone = False

        while curLength == limit:
            df = pd.read_sql("SELECT * FROM parentReply WHERE unix > {} AND parent NOT NULL AND score > 0 ORDER BY unix ASC LIMIT {}".format(lastUnix, limit), connection)
            lastUnix = df.tail(1)['unix'].values[0]
            curLength = len(df)
            if not testDone:
                with open("test.from",'a', encoding='utf8') as f:
                    for content in df['parent'].values:
                        f.write(content+'\n')
                with open("test.to",'a', encoding='utf8') as f:
                    for content in df['comment'].values:
                        f.write(content+'\n')
                
                testDone = True
            else:
                with open("train.from",'a', encoding='utf8') as f:
                    for content in df['parent'].values:
                        f.write(content+'\n')
                with open("train.to",'a', encoding='utf8') as f:
                    for content in df['comment'].values:
                        f.write(content+'\n')
            counter += 1
            if counter % 20 == 0:
                print(counter*limit, 'rows completed so far')