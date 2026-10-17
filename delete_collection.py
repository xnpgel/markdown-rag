"""
delete_collection.py
Connects to Milvus and delete the collection.
Usage: python delete_collection.py
"""

from pymilvus import connections, utility

import config

connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
if utility.has_collection(config.MILVUS_COLLECTION):
    utility.drop_collection(config.MILVUS_COLLECTION)
    print('Dropped old collection.')
