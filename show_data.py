"""
show_data.py
Connects to Milvus and shows collection information.
Usage: python show_data.py
"""

from pymilvus import connections, utility, Collection
import config

def show_collection_info():
    """Connect to Milvus and display collection details."""
    print(f"Connecting to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}...")
    
    try:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        print("Connected successfully.\n")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        return

    collection_name = config.MILVUS_COLLECTION

    # Check if collection exists
    if not utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return

    print(f"Collection '{collection_name}' exists.")

    # Load collection overview
    collection_count = utility.get_query_segment_info(collection_name)
    # Get number of entities
    collection = Collection(collection_name)
    num_entities = collection.num_entities
    
    print(f"Number of entities: {num_entities}")

    # Optional: Show schema
    schema = collection.schema
    print(f"Description: {schema.description}")
    print("Fields in schema:")
    for field in schema.fields:
        print(f"  - {field.name} ({field.dtype})")

    print("  - h1-h3: dynamic field.")

    print("  See this example:")
    first = collection.query(expr="h1 == 'SCANN'", limit=1, output_fields=["*"])
    print(first[0])

if __name__ == "__main__":
    show_collection_info()