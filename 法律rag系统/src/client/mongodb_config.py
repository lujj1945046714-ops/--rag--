# -*- coding: utf-8 -*-
import os
from pymongo import MongoClient

MONGO_HOST = os.environ.get("MONGO_HOST", "localhost")
MONGO_PORT = int(os.environ.get("MONGO_PORT", 27017))
MONGO_DB   = os.environ.get("MONGO_DB", "legal_rag")


class MongoConfig:
    _client = None
    _db = None

    @classmethod
    def initialize(cls):
        if cls._client is None:
            cls._client = MongoClient(host=MONGO_HOST, port=MONGO_PORT)
            cls._db = cls._client[MONGO_DB]

    @classmethod
    def get_db(cls):
        cls.initialize()
        return cls._db

    @classmethod
    def get_collection(cls, name: str):
        cls.initialize()
        return cls._db[name]

    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None


MongoConfig.initialize()
