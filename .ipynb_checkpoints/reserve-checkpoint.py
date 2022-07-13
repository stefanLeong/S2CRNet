import requests
import time
import os
from playsound import playsound
store_url = 'https://reserve-prime.apple.com/MO/zh_MO/reserve/A/stores.json'
product = 'https://reserve-prime.apple.com/MO/zh_MO/reserve/A/availability?iUP=N'
availability_url = 'https://reserve-prime.apple.com/MO/zh_MO/reserve/A/availability.json'

#iphone 13
# store_url = 'https://reserve-prime.apple.com/MO/zh_MO/reserve/D/stores.json'
# product = 'https://reserve-prime.apple.com/MO/zh_MO/reserve/D/availability?iUP=N'
# availability_url = 'https://reserve-prime.apple.com/MO/zh_MO/reserve/D/availability.json'

sound_alarm = 'alarm.mp3'

#iphone 13

stores = [('R672', '澳門銀河'), ('R697', '路氹金光大道')]
# stores = [('R320', '三里屯')]
product = 'MLTJ3ZA/A'  # iPhone 13 Pro 512G blue
# product = 'MGGU3CH/A'  # iPhone 12 128GB 黑色
product = 'MLTE3ZA/A'  # iPhone 13 Pro 256G blue
product = 'MLT93ZA/A'  # iPhone 113 Pro 256G 石墨色
# product = 'MGLA3CH/A'  # iPhone 12 Pro 128G 银色
# product = 'MLE23ZA/A' # iphone 13 pink
#
products = [
    ('MLTJ3ZA/A', ' iPhone 13 Pro 512G blue'),
    ('MLTE3ZA/A', 'iPhone 13 Pro 256G blue'),
    ('MLT93ZA/A', 'iPhone 13 Pro 256G black'),
]

# products = [
#     ('MLHC3ZA/A', 'iPhone 13  256G pink'),
# ]


print('店铺：', stores)
print('型号：', product)

s = requests.Session()
s.headers['User-Agent'] = 'Mozilla/5.0'

i = 0
while True:
    i += 1
    try:
        availability = s.get(availability_url, verify=False).json()
        for store in stores:
            for product in products:
                product_availability = availability['stores'][store[0]][product[0]]
                unlocked_state = product_availability['availability']['unlocked']
                print(i, '\t', store[1], '\t', product[1], '\t', product_availability)
                if unlocked_state:
                    print('say ' + store[1] + product[1])
                    playsound(sound_alarm)
                    break
                # print(unlocked_state)
    except Exception as e:
        print(i, '还没开始', e)

    time.sleep(1)

