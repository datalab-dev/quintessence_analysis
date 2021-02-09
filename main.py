""" 
Main script that runs LDA, word2vec models and updates the database

usage: poetry run python main.py
"""
import json

from quintessence.pipeline import EmbeddingsPipeline
from quintessence.pipeline import TopicModelPipeline

## load in json config file
args = json.load(open("config.json"))

## LDA
TopicModelPipeline(args)

## Embeddings
EmbeddingsPipeline(args)



