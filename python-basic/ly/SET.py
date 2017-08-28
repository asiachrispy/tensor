# -*- coding: utf-8 -*-

dict = {}
history_list = []

with open("weather_info.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        dict.setdefault(line.strip().split(",")[0], line.strip().split(",")[1])

while True:
    city = input("天气君欢迎你哟，想知道哪个城市的天气？")

    if city in dict:
        print(dict[city])
        history_list.append(city)

    elif city == "help":
        print(" 输入'help',查询帮助。 \n 输入'history',查询历史。 \n 输入'exit'，退出系统。")

    elif city == "history":
        for city in history_list:
            print(city, dict[city])

    elif city == "exit":
        exit(0)