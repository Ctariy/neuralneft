# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:42:02 2021

@author: SHER_CODE
"""

import telebot
import pandas as pd

bot = telebot.TeleBot('
                      ')

prediction = pd.read_csv(r'prediction.csv')
del prediction['Unnamed: 0']
prediction['Дата'] = pd.to_datetime(prediction['Дата'])
                     
@bot.message_handler(content_types=['text'])
def get_date(message):
    date = message.text
    try:
        if not date == 'Начать':
            if date.find('-') != -1:
                if len(date.split('-')) == 2:
                    date.replace(',', '.')
                    date = date.split('-')
                    date = [x.replace(' ', '') for x in date]
                    start = pd.to_datetime(date[0], format='%d.%m.%Y')
                    start = prediction.index[prediction['Дата'] == start]
                    
                    end = pd.to_datetime(date[1], format='%d.%m.%Y')
                    end = prediction.index[prediction['Дата'] == end]
                    
                    date = prediction.loc[start[0]:end[0]]
                    date.reset_index(drop = True, inplace = True)
                    for i in range(len(date)):
                        date.loc[i,'Дата'] = str(date['Дата'][i])[:len(str(date['Дата'][i]))-9]
                        
                        if len(str(round(date['Цена'][i], 2)).split('.')[1]) == 2:
                            date.loc[i, 'Цена'] = round(date['Цена'][i], 2)
                        else:
                            number = str(round(date['Цена'][i], 2))
                            number += '0'
                            print(number)
                            date.loc[i, 'Цена'] = number
        
                    text = ''
                    for i in range(len(date)):
                        text += str(date['Цена'][i]) +'\n'
        
                    if not text:
                        bot.send_message(message.from_user.id, 'На такой период данных нет.')
                    else:
                        bot.send_message(message.from_user.id, text)
                else:
                    pass
            else:
                index = pd.to_datetime(date)
                index = prediction.index[prediction['Дата'] == index]
                date = prediction.loc[index]
                date.reset_index(drop = True, inplace = True)
                text = round(date['Цена'], 2)
                if not text.any():
                    bot.send_message(message.from_user.id, 'На такой период данных нет.')
                else:
                    bot.send_message(message.from_user.id, text)
            
            print('Отправил!')
    except:
        bot.send_message(message.from_user.id, 
"""Отправьте дату или диапазон, чтобы получить прогноз!

Формат ввода - Д.М.Г или Д.М.Г - Д.М.Г""")
    
bot.polling(none_stop=True, interval=0)
