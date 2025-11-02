import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from free_llm import FreeLLM, get_free_llm
import networkx as nx
from tqdm import tqdm
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GraphDataProcessor:
    """Process and extract entities and relations from knowledge graph data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.entities = set()
        self.relations = set()
        self.triplets = []
        
    def load_data(self) -> List[Tuple[str, str, str, str]]:
        """Load data from the knowledge graph file"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    head, relation, tail = parts[0], parts[1], parts[2]
                    timestamp = parts[3] if len(parts) > 3 else None
                    data.append((head, relation, tail, timestamp))
                    self.entities.add(head)
                    self.entities.add(tail)
                    self.relations.add(relation)
                    self.triplets.append((head, relation, tail, timestamp))
        return data
    
    def extract_context_for_entities(self, llm_model: FreeLLM) -> Dict[str, str]:
        """Extract rich context for entities using LLM"""
        entity_contexts = {}
        
        print("Extracting context for entities using free LLM...")
        for entity in tqdm(list(self.entities)[:50]):  # Reduced limit for free model
            try:
                # Get related triplets for this entity
                related_triplets = [t for t in self.triplets if entity in [t[0], t[2]]][:3]
                
                context_prompt = f"""Entity: {entity}
Related events: {related_triplets}

Describe this entity in 1-2 sentences including its type and role."""
                
                response = llm_model.generate(context_prompt, max_length=200)
                entity_contexts[entity] = response
                
            except Exception as e:
                print(f"Error processing entity {entity}: {e}")
                entity_contexts[entity] = f"Entity: {entity}"
                
        return entity_contexts
    
    def extract_context_for_relations(self, llm_model: FreeLLM) -> Dict[str, str]:
        """Extract rich context for relations using LLM"""
        relation_contexts = {}
        
        print("Extracting context for relations using free LLM...")
        for relation in tqdm(list(self.relations)):
            try:
                # Get example triplets for this relation
                example_triplets = [t for t in self.triplets if t[1] == relation][:2]
                
                context_prompt = f"""Relation: {relation}
Examples: {example_triplets}

Explain this relation type in 1 sentence."""
                
                response = llm_model.generate(context_prompt, max_length=150)
                relation_contexts[relation] = response
                
            except Exception as e:
                print(f"Error processing relation {relation}: {e}")
                relation_contexts[relation] = f"Relation: {relation}"
                
        return relation_contexts

class MilvusManager:
    """Handle Milvus vector database operations"""
    
    def __init__(self):
        # Check if using Zilliz Cloud or local Milvus
        self.zilliz_uri = os.getenv('ZILLIZ_CLOUD_URI')
        self.zilliz_api_key = os.getenv('ZILLIZ_API_KEY')
        
        if self.zilliz_uri and self.zilliz_api_key:
            # Use Zilliz Cloud
            self.use_cloud = True
            self.host = None
            self.port = None
        else:
            # Use local Milvus
            self.use_cloud = False
            self.host = os.getenv('MILVUS_HOST', 'localhost')
            self.port = os.getenv('MILVUS_PORT', '19530')
        
        self.entity_collection_name = os.getenv('COLLECTION_NAME', 'graph_rag_entities')
        self.relation_collection_name = os.getenv('RELATION_COLLECTION_NAME', 'graph_rag_relations')
        self.embedding_model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
        
    def connect(self):
        """Connect to Milvus or Zilliz Cloud"""
        if self.use_cloud:
            connections.connect(
                "default",
                uri=self.zilliz_uri,
                token=self.zilliz_api_key
            )
            print("Connected to Zilliz Cloud")
        else:
            connections.connect("default", host=self.host, port=self.port)
            print("Connected to local Milvus")
        
    def create_entity_collection(self, dimension: int = 384):
        """Create collection for entities"""
        if utility.has_collection(self.entity_collection_name):
            utility.drop_collection(self.entity_collection_name)
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=100)
        ]
        
        schema = CollectionSchema(fields, "Entity collection for Graph RAG")
        collection = Collection(self.entity_collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        
        return collection
    
    def create_relation_collection(self, dimension: int = 384):
        """Create collection for relations"""
        if utility.has_collection(self.relation_collection_name):
            utility.drop_collection(self.relation_collection_name)
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="head_entity", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="relation", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="tail_entity", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        schema = CollectionSchema(fields, "Relation collection for Graph RAG")
        collection = Collection(self.relation_collection_name, schema)
        
        # Create index
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index("embedding", index_params)
        
        return collection
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def insert_entities(self, entities_data: Dict[str, str]):
        """Insert entities with embeddings into Milvus"""
        collection = Collection(self.entity_collection_name)
        
        entities = []
        contexts = []
        embeddings = []
        entity_types = []
        
        print("Generating embeddings for entities...")
        for entity, context in tqdm(entities_data.items()):
            # Combine entity name and context for embedding
            text_to_embed = f"{entity} {context}"
            embedding = self.embed_text(text_to_embed)
            
            entities.append(entity)
            contexts.append(context)
            embeddings.append(embedding)
            # Simple entity type classification (can be enhanced)
            entity_types.append(self._classify_entity_type(entity, context))
        
        data = [entities, contexts, embeddings, entity_types]
        collection.insert(data)
        collection.flush()
        print(f"Inserted {len(entities)} entities into Milvus")
    
    def insert_relations(self, triplets: List[Tuple], relation_contexts: Dict[str, str]):
        """Insert relations with embeddings into Milvus"""
        collection = Collection(self.relation_collection_name)
        
        head_entities = []
        relations = []
        tail_entities = []
        contexts = []
        embeddings = []
        timestamps = []
        
        print("Generating embeddings for relations...")
        for head, relation, tail, timestamp in tqdm(triplets):
            # Get relation context
            relation_context = relation_contexts.get(relation, relation)
            
            # Combine triplet information for embedding
            text_to_embed = f"{head} {relation} {tail} {relation_context}"
            embedding = self.embed_text(text_to_embed)
            
            head_entities.append(head)
            relations.append(relation)
            tail_entities.append(tail)
            contexts.append(relation_context)
            embeddings.append(embedding)
            timestamps.append(timestamp or "")
        
        data = [head_entities, relations, tail_entities, contexts, embeddings, timestamps]
        collection.insert(data)
        collection.flush()
        print(f"Inserted {len(triplets)} relations into Milvus")
    
    def _classify_entity_type(self, entity: str, context: str) -> str:
        """Simple entity type classification"""
        context_lower = context.lower()
        entity_lower = entity.lower()
        
        if any(word in context_lower for word in ['country', 'nation', 'state']):
            return 'Country'
        elif any(word in context_lower for word in ['president', 'minister', 'leader', 'official']):
            return 'Person'
        elif any(word in context_lower for word in ['organization', 'group', 'party', 'company']):
            return 'Organization'
        elif any(word in context_lower for word in ['city', 'capital', 'region']):
            return 'Location'
        else:
            return 'Other'

def main():
    """Main function to upload and process data"""
    
    # Initialize free LLM
    print("🤖 Initializing free LLM...")
    device = os.getenv('DEVICE', 'cpu')
    llm = get_free_llm("small_gpt", device)
    
    # Process graph data
    data_path = "datasets/icews14/train.txt"
    processor = GraphDataProcessor(data_path)
    
    print("Loading knowledge graph data...")
    triplets_data = processor.load_data()
    print(f"Loaded {len(triplets_data)} triplets")
    print(f"Found {len(processor.entities)} unique entities")
    print(f"Found {len(processor.relations)} unique relations")
    
    # Extract contexts using free LLM
    entity_contexts = processor.extract_context_for_entities(llm)
    relation_contexts = processor.extract_context_for_relations(llm)
    
    # Initialize Milvus
    milvus_manager = MilvusManager()
    milvus_manager.connect()
    
    # Create collections
    print("Creating Milvus collections...")
    milvus_manager.create_entity_collection()
    milvus_manager.create_relation_collection()
    
    # Insert data
    print("Inserting entities into Milvus...")
    milvus_manager.insert_entities(entity_contexts)
    
    print("Inserting relations into Milvus...")
    milvus_manager.insert_relations(processor.triplets, relation_contexts)
    
    # Save contexts for later use
    with open('entity_contexts.json', 'w') as f:
        json.dump(entity_contexts, f, indent=2)
    
    with open('relation_contexts.json', 'w') as f:
        json.dump(relation_contexts, f, indent=2)
    
    print("Data upload completed successfully!")

if __name__ == "__main__":
    main()
