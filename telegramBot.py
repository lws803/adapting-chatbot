#!/usr/bin/env python
# -*- coding: utf-8 -*-
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram
import logging
import time
import datetime
import random
import forecastio
from dateutil import tz
import requests
import boto3
import feedparser
import psutil
from googlesearch import search
import urllib.request
from bs4 import BeautifulSoup
from unidecode import unidecode
import argparse

from stat_parser import Parser as TreeParser
tree_parser = TreeParser()

parser = argparse.ArgumentParser()

parser.add_argument('--telegram_api', type=str)
parser.add_argument('--weather_api', type=str)
parser.add_argument('--youtube_api', type=str)


args = parser.parse_args()

# TODO: Implement identity and saving data for the telegram chatbot here

threshold = 0.8
learningInput = {}
learningResponse = {}
randomNum = 0

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    """Send a message when the command /start is issued."""
    update.message.reply_text('Onegaishimashka\n\ntype /help to see available commands')


def help(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    """Send a message when the command /help is issued."""
    update.message.reply_text('/cancel - cancel learn request \n')


def respond(user_input, update, bot):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    # TODO: If threshold is small, then we trigger storing in message queue

    # response = chatbot.get_response(user_input)

    # if (response.confidence < threshold and not(str(update.message.from_user.username) in learningInput)):
    #     update.message.reply_text("What talking you? what should I say?")

    #     learningInput[str(update.message.from_user.username)] = user_input
    #     learningResponse[str(update.message.from_user.username)] = response

    # elif (str(update.message.from_user.username) in learningInput):
    #     corrected_response = learningResponse[str(update.message.from_user.username)]
    #     corrected_response.text = user_input.text

    #     chatbot.learn_response(corrected_response, learningInput[str(update.message.from_user.username)])
    #     #chatbot.storage.add_to_conversation(CONVERSATION_ID, user_input, corrected_response)
    #     update.message.reply_text("Roger. I'll learn from you")

    #     learningInput.pop(str(update.message.from_user.username), None)
    #     learningResponse.pop(str(update.message.from_user.username), None)
    # else:
    #     update.message.reply_text(response.text)


def chinchin(bot, update):
    if (update.message.chat_id == -288581501):
        #user_input = Statement("")
        text = update.message.text.replace("@chinchin519bot ", "")
        user_input = text  # Needed to turn str input into a statement
        respond(user_input, update, bot)

        #print update.message.chat_id
    else:
        update.message.reply_text("Please talk to @chinchin519bot instead..") 


def learn(bot, update, args):
    user_input = ''
    # update.message.reply_text("What shd be my response? I'm here to learn and replace")
    # for items in args:
    #     user_input += str(items)
    #     user_input += " "
    # print (user_input)
    # learningInput[str(update.message.from_user.username)] = Statement(user_input)
    # learningResponse[str(update.message.from_user.username)] = Statement(user_input)


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def setThresh(bot, update, args):
    global threshold
    threshold = float(str(args[0]))
    update.message.reply_text("Threshold set!")

def showThresh(bot, update):
    update.message.reply_text(threshold)

def cancelLearn(bot, update): 
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING) # Added this to make it more natural
    learningInput.pop(str(update.message.from_user.username), None)
    learningResponse.pop(str(update.message.from_user.username), None)
    update.message.reply_text("Aiyo play play play")


def getWeather(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING) # Added this to make it more natural
    api_key = args.weather_api
    lat = 1.353435
    lng = 103.852668  # Singapore
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Singapore')
    curr_time = datetime.datetime.utcnow()
    curr_time = curr_time.replace(tzinfo=from_zone)
    curr_time = curr_time.astimezone(to_zone)
    forecast = forecastio.load_forecast(api_key, lat, lng, time=time)
    summary = forecast.currently().summary
    temperature = "Average temperature: " + str(forecast.currently().temperature) + " C"
    update.message.reply_text(summary + "\n" + temperature)


def getTime(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING) 
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('Singapore')
    time = datetime.datetime.utcnow()
    time = time.replace(tzinfo=from_zone)
    time = time.astimezone(to_zone)
    update.message.reply_text(time.strftime("%c"))


def nekos(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING) # Added this to make it more natural
    s = requests.Session()
    r = s.get('https://nekos.life/api/neko')
    data = r.json()
    update.message.reply_text("Here's a neko for you~\n" + data['neko'])
    #bot.send_photo(chat_id=update.message.chat_id, photo=data['neko'])

def getCat(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING) # Added this to make it more natural
    iserror = True
    retries = 0
    while (iserror and retries < 10):
        try:
            s = requests.Session()
            r = s.get('http://aws.random.cat/meow')
            data = r.json()
            iserror = False
            update.message.reply_text("Here's a kitty for you~\n" + data['file'])
        except ValueError:
            print("Error, trying again...")
            retries += 1
    if (iserror):
        print("Error sending")
        update.message.reply_text("Error, try again :3")

    #bot.send_photo(chat_id=update.message.chat_id, photo="http://thecatapi.com/api/images/get?format=src")
    #update.message.reply_text("Here's a kitty for you~\n" + "http://thecatapi.com/api/images/get?format=src")


def getCorgi(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)  # Added this to make it more natural
    s = requests.Session()
    r = s.get('https://dog.ceo/api/breed/corgi/cardigan/images/random')
    data = r.json()
    update.message.reply_text("Here's a corgi for you~\n" + data['message'])
    #bot.send_photo(chat_id=update.message.chat_id, photo=data['file'])


def getJames(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)  # Added this to make it more natural
    bot.send_photo(chat_id=update.message.chat_id, photo=open('james/james.jpg', 'rb'))

def getGab(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)  # Added this to make it more natural
    randomNum = random.randint(0, 4)
    bot.send_photo(chat_id=update.message.chat_id, photo=open('gab/' + str(randomNum) + '.jpeg', 'rb'))

def getJulian(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)  # Added this to make it more natural
    randomNum = random.randint(0, 7)
    bot.send_photo(chat_id=update.message.chat_id, photo=open('julian/' + str(randomNum) + '.jpeg', 'rb'))

def getSort(bot, update, args):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)  # Added this to make it more natural
    if (len(args) == 0):
        update.message.reply_text("Give me what sort you want la... use /sort bubblesort for example")
        return
    if (str(args[0]).lower() == "bubblesort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/bubbleSort.txt', 'rb'))
    elif (str(args[0]).lower() == "mergesort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/mergeSort.txt', 'rb'))        
    elif (str(args[0]).lower() == "insertionsort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/insertionSort.txt', 'rb'))                
    elif (str(args[0]).lower() == "selectionsort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/selectionSort.txt', 'rb'))                
    elif (str(args[0]).lower() == "quicksort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/quickSort.txt', 'rb'))                        
    elif (str(args[0]).lower() == "countingsort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/countingSort.txt', 'rb'))                
    elif (str(args[0]).lower() == "radixsort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/radixSort.txt', 'rb'))                
    elif (str(args[0]).lower() == "heapsort"):
        bot.send_document(chat_id=update.message.chat_id, document=open('./sortingAlgos/heapSort.txt', 'rb'))                        
    else: 
        update.message.reply_text("No such sort found in my database...")


def getYoutubeLink(bot, update):
    channels = ['PLzJj8MimD4DEV_19cvlG5cQyZ0pnF2d1i', 'PLPPpv0bD5M6ia-OowntB1ev6d4tQXKfdq', 'PLzJj8MimD4DHIyxcWS86vTkT37NhASzSo']
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING) # Added this to make it more natural
    myLink = channels[random.randint(0, 1)]
    playlist_id = str(myLink)
    links = []
    token = ''  # to get ID to next page  CDIQAA

    # https://stackoverflow.com/questions/14173428/how-to-change-page-results-with-youtube-data-api-v3 - reference
    s = requests.Session()
    r = s.get('https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&playlistId=' + 
        playlist_id + '&key=' + args.youtube_api)
    data = r.json()

    while 'nextPageToken' in data and len(data) < 200:
        token = data['nextPageToken']

        r = s.get('https://www.googleapis.com/youtube/v3/playlistItems?pageToken=' + token + 
        '&part=snippet&maxResults=50&playlistId=' + playlist_id + '&key=' + args.youtube_api)
        data = r.json()
        for items in data['items']:
            links.append('https://www.youtube.com/watch?v=' + items['snippet']['resourceId']['videoId'])

    r = s.get('https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&playlistId=' +
        playlist_id + '&key=' + args.youtube_api)
    data = r.json()
    # Add the remaining links
    for items in data['items']:
        links.append('https://www.youtube.com/watch?v=' + items['snippet']['resourceId']['videoId'])
    #print "num items: " + str(len(links))
    update.message.reply_text(links[random.randint(0, len(links) - 1)])



def getKattisQuestions(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    # Added this to make it more natural
    d = feedparser.parse('https://open.kattis.com/rss/new-problems')
    size = len(d.entries)
    randNum = random.randint(0, size - 1)
    link = d.entries[randNum].link
    title = d.entries[randNum].title
    update.message.reply_text(title + "\n" + link)


def getStatus(bot, update): 
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    # Added this to make it more natural
    ec2 = boto3.resource('ec2')
    for status in ec2.meta.client.describe_instance_status()['InstanceStatuses']:
        update.message.reply_text(
            "Status: " + status['SystemStatus']['Status']
            + "\n" + "CPU usage: " + str(psutil.cpu_percent(interval=1))
        )


def google_scrape(url):
    thepage = urllib.request.urlopen(url)
    soup = BeautifulSoup(thepage, "html.parser")
    return soup

def google_search(bot, update, args):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    user_input = ""
    for items in args:
        user_input += str(unidecode(items))
        user_input += " "
    print(user_input)
    query = user_input
    for url in search(query, stop=20):
        a = google_scrape(url)
        if (a.title is not None):
            update.message.reply_text(a.title.text + "\n" + url)
            break


def youtube_search(bot, update, args):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    user_input = ""
    for items in args:
        user_input += str(unidecode(items))
        user_input += " "
    query = user_input
    if (len(args) == 0):
        # Search trending
        payload = {
            'chart': "mostPopular", 
            'part': 'snippet',
            'maxResults': 20,
            'key': args.youtube_api
        }

        s = requests.Session()
        r = s.get('https://www.googleapis.com/youtube/v3/videos', params=payload)
        data = r.json()

        id_array = []
        for items in data['items']:
            id_array.append(items['id'])

        update.message.reply_text('https://www.youtube.com/watch?v=' + id_array[random.randint(0,len(id_array) - 1)])

    else:
        payload = {
            'q': query,
            'part': 'snippet',
            'maxResults': 1,
            'key': args.youtube_api
        }

        s = requests.Session()
        r = s.get('https://www.googleapis.com/youtube/v3/search', params=payload)
        data = r.json()
        for items in data['items']:
            update.message.reply_text('https://www.youtube.com/watch?v=' + items['id']['videoId'])
            break

def searchUrban(bot, update, args):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    user_input = ""
    for items in args:
        user_input += str(unidecode(items))
        user_input += " "
    query = user_input
    if (len(args) != 0):
        payload = {
            'term': query
        }

        s = requests.Session()
        r = s.get('http://api.urbandictionary.com/v0/define', params=payload)
        data = r.json()
        definitions = []
        for item in data["list"]:
            definitions.append(item["definition"] + "\n" + item["permalink"])

    update.message.reply_text(definitions[random.randint(0, len(definitions) - 1)])

def buhuireply(bot, update):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    #bot.send_photo(chat_id=update.message.chat_id, photo=open('buhuireply.mp4', 'rb'))
    bot.send_document(chat_id=update.message.chat_id, document=open('buhuireply.mp4', 'rb'))


def parseTree(bot, update, args):
    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)

    user_input = ""
    for items in args:
        user_input += str(unidecode(items))
        user_input += " "


    update.message.reply_text(str(tree_parser.parse(user_input)))


def main():

    #chatbot.train("chatterbot.corpus.chinchin")

    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(args.telegram_api)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("learn", learn, pass_args=True))
    dp.add_handler(CommandHandler("threshold", setThresh, pass_args=True))
    dp.add_handler(CommandHandler("showthresh", showThresh))
    dp.add_handler(CommandHandler("cancel", cancelLearn))
    dp.add_handler(CommandHandler("weather", getWeather))
    dp.add_handler(CommandHandler("time", getTime))
    dp.add_handler(CommandHandler("neko", nekos))
    dp.add_handler(CommandHandler("cat", getCat))
    dp.add_handler(CommandHandler("corgi", getCorgi))
    dp.add_handler(CommandHandler("status", getStatus))
    dp.add_handler(CommandHandler("james", getJames))
    #dp.add_handler(CommandHandler("gabriel", getGab))
    dp.add_handler(CommandHandler("julian", getJulian))
    dp.add_handler(CommandHandler("kattis", getKattisQuestions))
    dp.add_handler(CommandHandler("sort", getSort, pass_args=True))
    dp.add_handler(CommandHandler("music", getYoutubeLink))
    dp.add_handler(CommandHandler("music", getYoutubeLink))
    dp.add_handler(CommandHandler("google", google_search, pass_args=True))
    dp.add_handler(CommandHandler("youtube", youtube_search, pass_args=True))
    dp.add_handler(CommandHandler("buhuireply", buhuireply))
    dp.add_handler(CommandHandler("search", searchUrban, pass_args=True))
    dp.add_handler(CommandHandler("parse", parseTree, pass_args=True))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, chinchin))


    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    print("All systems started")

    updater.idle()


if __name__ == '__main__':
    main()
