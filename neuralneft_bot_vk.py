# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 23:08:39 2020

@author: SHER_CODE
"""

from vk_api import VkApi
import random
import pandas as pd
import sys
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType

def send(user_id, text):
    random_id = random.randint(-2147483648, +2147483648)
    vk.messages.send(user_id=user_id, random_id = random_id, message=text)

token = 'f691667351200663e3d7dce46df80478d8d9574c616a8c423d2efac2735a20f7d45e1dcf9248fb58121a6'
vk_session = VkApi(token=token)
longpoll = VkBotLongPoll(vk_session, '198903774')
vk = vk_session.get_api()

prediction = pd.read_csv(r'prediction.csv')
del prediction['Unnamed: 0']
prediction['Дата'] = pd.to_datetime(prediction['Дата'])
print(prediction)

for event in longpoll.listen():
    if event.type == VkBotEventType.MESSAGE_NEW:
        print('GOTCHA')
        if event.object['message']['text'].lower() == '/restart' and (str(event.object['message']['from_id']) == '179481950' or str(event.object['message']['from_id']) == '189751408'):
            vk.messages.send(
                user_id=event.user_id, random_id = random.randint(-2147483648, +2147483648),message='Бот перезагружается'
                )
            sys.exit(0)  
        elif event.object['message']['text'].lower() == 'начать':
            
            user_id = event.object['message']['from_id']
            send(user_id, 'Отправьте дату или диапазон, чтобы получить прогноз!\n\nФормат ввода - Д.М.Г или Д.М.Г - Д.М.Г')
        else:
            response = event.object['message']['text'].lower()
            user_id = event.object['message']['from_id']
            if response.find('-') != -1:
                if len(response.split('-')) == 2:
                    response.replace(',', '.')
                    response = response.split('-')
                    response = [x.replace(' ', '') for x in response]
                    start = pd.to_datetime(response[0], format='%d.%m.%Y')
                    start = prediction.index[prediction['Дата'] == start]
                    
                    end = pd.to_datetime(response[1], format='%d.%m.%Y')
                    end = prediction.index[prediction['Дата'] == end]
                    
                    result = prediction.loc[start[0]:end[0]]
                    result.reset_index(drop = True, inplace = True)
                    for i in range(len(result)):
                        result.loc[i,'Дата'] = str(result['Дата'][i])[:len(str(result['Дата'][i]))-9]
                        
                        if len(str(round(result['Цена'][i], 2)).split('.')[1]) == 2:
                            result.loc[i, 'Цена'] = round(result['Цена'][i], 2)
                        else:
                            number = str(round(result['Цена'][i], 2))
                            number += '0'
                            print(number)
                            result.loc[i, 'Цена'] = number

                    message = ''
                    for i in range(len(result)):
                        message += str(result['Цена'][i]) +'\n'

                    if not message:
                        send(user_id, 'На такой период данных нет.')
                    else:
                        send(user_id, message)
                else:
                    pass
            else:
                index = pd.to_datetime(response)
                index = prediction.index[prediction['Дата'] == index]
                result = prediction.loc[index]
                result.reset_index(drop = True, inplace = True)
                message = round(result['Цена'], 2)
                if not message.any():
                    send(user_id, 'На такой период данных нет.')
                else:
                    send(user_id, message)
            
            print('Отправил!')