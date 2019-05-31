from datetime import datetime
import os

def make_folder_with_time(path='.'):
    dic = {1:'Jan', 2:'Feb', 3:"Mar", 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    now = datetime.now()
    dt = '{}{}_{}-{}'.format(dic[now.month], now.day, now.hour, now.minute)
    os.makedirs(os.path.join(path, dt))
    return dt