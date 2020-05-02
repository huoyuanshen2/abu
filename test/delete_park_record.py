#!/usr/bin/python
# coding:utf-8
#TOB项目定时调度任务。
import threading
from time import ctime
import pymysql
import time
import datetime
import logging
import calendar

wait_time = 0.1
start=time.strftime("%Y-%m-%d",time.localtime(time.time()))


def insert_run(datestart=None, delete=False, rule=None,cur=None,db=None):

    db_name = rule[1]
    tb_name = rule[2]
    column_name = rule[3]
    column_type = rule[4]
    status = rule[5]
    keep_day = rule[6]
    copy_day = rule[7]

    dateend = datestart + datetime.timedelta(days=1)
    year_name = datestart.strftime('%Y')

    crate_table_sql = '''CREATE TABLE if not exists {0}.{1}_{2} LIKE {0}.{1};'''.format(db_name, tb_name, year_name)
    cur.execute(crate_table_sql)

    # 删除sql。执行前是否需要再次确认？需要。遇到异常，将记录存入数据库。
    while datestart < dateend:
        datestart += datetime.timedelta(days=1)
        datestart_temp = datestart + datetime.timedelta(days=-1)
        dateend_temp = datestart

        year_name = datestart_temp.strftime('%Y')

        start_time = datestart_temp.strftime('%Y-%m-%d 00:00:00')
        end_time = dateend_temp.strftime('%Y-%m-%d 00:00:00')
        start_date = datestart_temp.strftime('%Y-%m-%d')

        sql = []
        sql.append('''
                SELECT count(*)  FROM {db_name}.{tb_name}  WHERE {column_name}  >= '{start_time}' and {column_name}  < '{end_time}';
                 ''')
        sql.append('''
        INSERT ignore INTO {db_name}.{tb_name}_{year_name} SELECT *  FROM {db_name}.{tb_name}  WHERE  {column_name}  
         >= '{start_date} {start_hour}:00:00' and {column_name} < '{start_date} {end_hour}:00:00';
         ''')
        sql.append('''
                SELECT count(*)  FROM {db_name}.{tb_name}_{year_name}  WHERE {column_name}  >= '{start_time}' and {column_name}  < '{end_time}';
                 ''')
        sql.append('''
        delete  FROM {db_name}.{tb_name}  WHERE {column_name} >= '{start_date} {start_hour}:00:00' and {column_name} < '{start_date} {end_hour}:00:00' limit 500;
        ''')

        sel_sql1 = sql[0].format(db_name=db_name, tb_name=tb_name, year_name=year_name, column_name=column_name,
                                 start_time=start_time, end_time=end_time)
        insert_sql = sql[1]
        sel_sql2 = sql[2].format(db_name=db_name, tb_name=tb_name, year_name=year_name, column_name=column_name,
                                 start_time=start_time, end_time=end_time)
        del_sql = sql[3]
        print(sel_sql1)
        print(sel_sql2)

        cur.execute(sel_sql1)
        result = cur.fetchone()
        sel_num = result[0]
        if (sel_num == 0):
            continue
        hour_list = ['{num:02d}'.format(num=i) for i in range(24)]
        for hour in hour_list:
            insert_sql_hour = insert_sql.format(db_name=db_name, tb_name=tb_name,
                                                year_name=year_name, column_name=column_name, start_time=start_time,
                                                end_time=end_time, start_date=start_date, start_hour=hour,
                                                end_hour=str(int(hour) + 1))
            cur.execute(insert_sql_hour)
            db.commit()
            time.sleep(wait_time)
        totalDay = getTotalday(start_time)
        if totalDay <= copy_day: #基于年初的数据，需要冗余迁移到上一年的表中：year_name=year_name-1
            for hour in hour_list:
                insert_sql_hour = insert_sql.format(db_name=db_name, tb_name=tb_name,
                                                    year_name=year_name-1, column_name=column_name, start_time=start_time,
                                                    end_time=end_time, start_date=start_date, start_hour=hour,
                                                    end_hour=str(int(hour) + 1))
                cur.execute(insert_sql_hour)
                db.commit()
                time.sleep(wait_time)

        cur.execute(sel_sql2)
        result2 = cur.fetchone()
        sel_num_history = result2[0]
        print("sel_num,sel_num_history", sel_num, sel_num_history)
        if (sel_num == sel_num_history) and delete == True:
            for hour in hour_list:
                resultInt = 1;
                while resultInt > 0:
                    del_sql_hour = del_sql.format(db_name=db_name, tb_name=tb_name,
                                                  year_name=year_name, column_name=column_name, start_time=start_time,
                                                  end_time=end_time, start_date=start_date, start_hour=hour,
                                                  end_hour=str(int(hour) + 1))
                    print(del_sql_hour)
                    resultInt = cur.execute(del_sql_hour)
                    db.commit()
                    trace_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                    message = "TRACE_ID:%s, SQL:%s, RESULT:%s" % (trace_id, del_sql_hour, resultInt)
                    time.sleep(wait_time)
                    logging.info(message)
        time.sleep(wait_time)
    db.commit()

def ordinary_insert():
    # db = pymysql.connect(host='rm-2ze3z0062odds5527.mysql.rds.aliyuncs.com',  # TOB商用环境内网
    # user = 'zhht_yunwei',
    # passwd = '@5A02F0F3F6D06A5E',
    db=pymysql.connect(host='rm-2ze77mm1d0lw6xo7nko.mysql.rds.aliyuncs.com',  # TOG test环境
    user='huoyuanshen',
    passwd='857911Hys',
    port = 3306,
    db = 'dba_test',
    #db='ysczhhtdb',
    charset = "utf8")
    cur = db.cursor()

    date_now = datetime.datetime.strptime(start, '%Y-%m-%d')
    now_year_name = date_now.strftime('%Y')

    rule_sql = '''SELECT *  FROM `z_keep_rule` where status=1 ;'''
    cur.execute(rule_sql)
    result = cur.fetchall()
    for rule in result:
        db_name = rule[1]
        tb_name = rule[2]
        column_name = rule[3]
        column_type = rule[4]
        status = rule[5]
        keep_day = rule[6]
        copy_day = rule[7]
        # crate_table_sql = '''CREATE TABLE if not exists {0}.{1}_{2} LIKE {0}.{1};'''.format(db_name, tb_name, now_year_name)
        # cur.execute(crate_table_sql)

        datestart = date_now + datetime.timedelta(days=-keep_day)
        insert_run(datestart=datestart,delete=False,rule=rule,cur=cur,db=db)

        copy_datestart = date_now + datetime.timedelta(days=-(keep_day-copy_day))
        insert_run(datestart=copy_datestart, delete=False, rule=rule,cur=cur,db=db)


    print(datestart)
    db.close()  # 关闭连接

print('start:', ctime())
start2 = time.time()

if __name__ == '__main__':
    ordinary_insert()

seconds = time.time() - start2
print('end:', ctime())
print("{func}函数每条数数据写入耗时{sec}秒".format(func='ordinary_insert',  sec=seconds))

def getTotalday(dataTime):
    year = int(dataTime.strftime('%Y'))
    month = int(dataTime.strftime('%m'))
    day = int(dataTime.strftime('%d'))
    totalday = 0
    if calendar.isleap(year):
        days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
       days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(1, 13):
        if i == month:
            for j in range(1,i):
                totalday = totalday + days[j-1]
    totalday = totalday + day
    return totalday
