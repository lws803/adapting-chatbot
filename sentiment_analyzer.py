from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--file', type=str, help='file location')

args = parser.parse_args()

if args.file is None:
    exit(1)


f = open(args.file,"r")
text = f.read()
# text = ""

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


sentiment_analyzer_scores(text)
