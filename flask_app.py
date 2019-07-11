from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from aiohttp import web
import simplejson as json
import aiohttp_cors

import torch
import torch.nn as nn
import os
import argparse
from utils import (MAX_LENGTH,
                normalizeString,
                indexesFromSentence)
from dataloader import DataLoader, SAVE_DIR, CORPUS_NAME
from model import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder

parser = argparse.ArgumentParser()

parser.add_argument('--hidden_size', default=500, type=int)
parser.add_argument('--model_name', default='cb_model', type=str)
parser.add_argument('--attn_model', default='dot', type=str, help='general, concat or dot',
                    choices=['dot', 'general', 'concat'])
parser.add_argument('--encoder_n_layers', default=2, type=int)
parser.add_argument('--decoder_n_layers', default=2, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--checkpoint_iter', default=26000, type=int)
parser.add_argument('--threshold', default=0.2, type=float)


args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

dl = DataLoader()
voc, pairs = dl.generate_voc_pairs()


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    score = 1
    for item in scores:
        score *= item
    print("scores:", float(score))
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words, float(score)


loadFilename = os.path.join(SAVE_DIR, args.model_name, CORPUS_NAME,
                        '{}-{}_{}'.format(args.encoder_n_layers, args.decoder_n_layers, args.hidden_size),
                        '{}_checkpoint.tar'.format(args.checkpoint_iter))

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename, map_location='cpu')
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, args.hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(args.hidden_size, embedding, args.encoder_n_layers, args.dropout)
decoder = LuongAttnDecoderRNN(args.attn_model, embedding, args.hidden_size, voc.num_words, args.decoder_n_layers, args.dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


threshold = 0.7
learningInput = {}
learningResponse = {}
randomNum = 0

encoder.eval()
decoder.eval()
# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)


app = web.Application()

async def handle(request):
    my_input = request.query['input']
    input_sentence = normalizeString(my_input)
    # Evaluate sentence
    output_words, score = evaluate(encoder, decoder, searcher, voc, input_sentence)
    # Format and print response sentence
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]

    response_obj = {'status': 'success', "response": ' '.join(output_words), "confidence": score}

    return web.Response(text=json.dumps(response_obj), status=200)



app.router.add_route("GET", "/input", handle)

# Configure default CORS settings.
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
})

# Configure CORS on all routes.
for route in list(app.router.routes()):
    cors.add(route)


web.run_app(app, port=5000)
