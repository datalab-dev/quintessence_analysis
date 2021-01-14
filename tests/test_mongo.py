"""
This script tests the methods of the Mongo class to ensure it reads and writes
the data without issues from a properly formatted mongodb database.

Mongo database needs
docs.meta
docs.lemma
"""
import pytest

from quintessence.mongo import Mongo

filepath = "./test_credentials.json"

def test_connection():
    con = Mongo(filepath)
    collections = con.db.list_collection_names(include_system_collections=False)
    assert(all(x in collections for x in ['docs.meta', 'docs.lemma']))

def test_get_metadata():
    con = Mongo(filepath)
    meta = con.get_metadata()
    keys = list(meta[0].keys())
    assert(all(x in keys for x in ["_id", "File_ID", "VID", "EEBO_Citation",
        "Proquest_ID", "STC_ID", "ESTC_ID", "Title", "Location", "Publisher",
        "Author", "Keywords", "Language", "Date"]))

def test_get_topic_model_data():
    con = Mongo(filepath)
    ids,docs = con.get_topic_model_data()
    assert(type(ids[0]) == int and type(docs[0] == str))

def test_write_topic_model_data():
    # TODO:
    pass

def test_get_embed_data():
    # TODO:
    pass

def test_write_embed_model_data():
    # TODO:
    pass
