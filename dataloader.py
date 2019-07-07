from io import open
from utils import (normalizeString,
    indexesFromSentence,
    zeroPadding,
    filterPairs,
    binaryMatrix,
    printLines,
    extractSentencePairs,
    trimRareWords,
    MIN_COUNT)
from vocab import Voc
import torch
import os
import codecs
import csv

CORPUS_NAME = "cornell movie-dialogs corpus"
CORPUS = os.path.join("data", CORPUS_NAME)
SAVE_DIR = os.path.join("data", "save")
DELIMETER = '\t'

# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split(DELIMETER)] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

class DataLoader:

    def wash_data(self, datafile):
        # Unescape the delimiter
        delimiter = str(codecs.decode(DELIMETER, "unicode_escape"))

        # Initialize lines dict, conversations list, and field ids
        lines = {}
        conversations = []
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

        # Load lines and process conversations
        print("\nProcessing corpus...")
        lines = loadLines(os.path.join(CORPUS, "movie_lines.txt"), MOVIE_LINES_FIELDS)
        print("\nLoading conversations...")
        conversations = loadConversations(os.path.join(CORPUS, "movie_conversations.txt"),
                                        lines, MOVIE_CONVERSATIONS_FIELDS)

        # Write new csv file
        print("\nWriting newly formatted file...")
        with open(datafile, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
            for pair in extractSentencePairs(conversations):
                writer.writerow(pair)


    def generate_voc_pairs(self):
        printLines(os.path.join(CORPUS, "movie_lines.txt"))
        # Define path to new file
        datafile = os.path.join(CORPUS, "formatted_movie_lines.txt")

        if not (os.path.exists(datafile)):
            self.wash_data(datafile)

        # Print a sample of lines
        print("\nSample lines from file:")
        printLines(datafile)


        # Load/Assemble voc and pairs
        voc, pairs = loadPrepareData(CORPUS, CORPUS_NAME, datafile, SAVE_DIR)

        # Trim voc and pairs
        pairs = trimRareWords(voc, pairs, MIN_COUNT)
        return voc, pairs


if __name__ == "__main__":
    dl = DataLoader()
    voc, pairs = dl.generate_voc_pairs()
