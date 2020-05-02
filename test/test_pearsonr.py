import pymysql
import pandas as pd
from sqlalchemy import create_engine

# pymysql.install_as_MySQLdb()  如果你想用MySQLdb 的话把这行代码打开就行, 建议用pymysql


connect = create_engine(
    f'mysql+pymysql://dba_4_update:857911Hys@rm-2ze68jb334ufy64rano.mysql.rds.aliyuncs.com:3306/dba_test?charset=utf8')

lis = [
    {"datetime": 20190516, "type": 1, "count": 869},
    {"datetime": 20190517, "type": 1, "count": 869},
]

df = pd.DataFrame(lis)
# 如果想要自动建表的话把if_exists的值换为replace, 建议自己建表
df.to_sql("com_count_temp02", connect, if_exists='append', index=False)