import os
import json
from typing import List, Dict, Tuple, Any
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from free_llm import FreeLLM, get_free_llm
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GraphRAGQueryEngine:
    """Query engine for Graph RAG system"""
    
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
        
        # Initialize free LLM
        device = os.getenv('DEVICE', 'cpu')
        self.llm = get_free_llm("small_gpt", device)
        
        # Connect to Milvus or Zilliz Cloud
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
        
        # Load collections
        self.entity_collection = Collection(self.entity_collection_name)
        self.relation_collection = Collection(self.relation_collection_name)
        
        # Load collections into memory for better performance
        self.entity_collection.load()
        self.relation_collection.load()
        
        print("GraphRAG Query Engine initialized successfully")
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query"""
        embedding = self.embedding_model.encode(query)
        return embedding.tolist()
    
    def search_entities(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for relevant entities"""
        query_embedding = self.embed_query(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = self.entity_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["entity", "context", "entity_type"]
        )
        
        entities = []
        for hit in results[0]:
            entities.append({
                "entity": hit.entity.get("entity"),
                "context": hit.entity.get("context"),
                "entity_type": hit.entity.get("entity_type"),
                "score": hit.score
            })
        
        return entities
    
    def search_relations(self, query: str, top_k: int = 15) -> List[Dict]:
        """Search for relevant relations"""
        query_embedding = self.embed_query(query)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = self.relation_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["head_entity", "relation", "tail_entity", "context", "timestamp"]
        )
        
        relations = []
        for hit in results[0]:
            relations.append({
                "head_entity": hit.entity.get("head_entity"),
                "relation": hit.entity.get("relation"),
                "tail_entity": hit.entity.get("tail_entity"),
                "context": hit.entity.get("context"),
                "timestamp": hit.entity.get("timestamp"),
                "score": hit.score
            })
        
        return relations
    
    def extract_knowledge_subgraph(self, query: str, entity_top_k: int = 10, relation_top_k: int = 15) -> Dict:
        """Extract relevant knowledge subgraph for the query"""
        
        # Search for relevant entities and relations
        relevant_entities = self.search_entities(query, entity_top_k)
        relevant_relations = self.search_relations(query, relation_top_k)
        
        # Filter relations that involve relevant entities
        entity_names = {entity["entity"] for entity in relevant_entities}
        filtered_relations = []
        
        for relation in relevant_relations:
            if (relation["head_entity"] in entity_names or 
                relation["tail_entity"] in entity_names):
                filtered_relations.append(relation)
        
        # Also add entities mentioned in filtered relations
        additional_entities = set()
        for relation in filtered_relations:
            additional_entities.add(relation["head_entity"])
            additional_entities.add(relation["tail_entity"])
        
        # Search for additional entities if needed
        for entity_name in additional_entities:
            if not any(e["entity"] == entity_name for e in relevant_entities):
                # Quick search for this specific entity
                entity_results = self.search_entities(entity_name, 1)
                if entity_results and entity_results[0]["score"] > 0.8:
                    relevant_entities.append(entity_results[0])
        
        return {
            "entities": relevant_entities,
            "relations": filtered_relations
        }
    
    def generate_context_summary(self, knowledge_subgraph: Dict) -> str:
        """Generate a comprehensive context summary from the knowledge subgraph"""
        
        entities = knowledge_subgraph["entities"]
        relations = knowledge_subgraph["relations"]
        
        # Create entity descriptions
        entity_descriptions = []
        for entity in entities[:8]:  # Limit to avoid token overflow
            entity_descriptions.append(
                f"- {entity['entity']} ({entity['entity_type']}): {entity['context']}"
            )
        
        # Create relation descriptions
        relation_descriptions = []
        for relation in relations[:12]:  # Limit to avoid token overflow
            rel_desc = f"- {relation['head_entity']} → {relation['relation']} → {relation['tail_entity']}"
            if relation['timestamp']:
                rel_desc += f" (Date: {relation['timestamp']})"
            if relation['context']:
                rel_desc += f" | Context: {relation['context']}"
            relation_descriptions.append(rel_desc)
        
        context_summary = f"""
RELEVANT ENTITIES:
{chr(10).join(entity_descriptions)}

RELEVANT RELATIONSHIPS:
{chr(10).join(relation_descriptions)}
"""
        
        return context_summary
    
    def generate_answer(self, query: str, context_summary: str) -> str:
        """Generate final answer using free LLM with the retrieved context"""
        
        prompt = f"""You are an expert analyst. Answer the following query based on the knowledge graph information provided.

Query: {query}

Knowledge Graph Context:
{context_summary}

Based on the above context, provide a comprehensive answer that:
1. Directly addresses the query
2. Uses specific information from the entities and relationships
3. Explains relevant connections or patterns
4. Mentions timing when available

Answer:"""
        
        # Generate response using free LLM
        response = self.llm.generate(prompt, max_length=400, temperature=0.3)
        return response.strip()
    
    def query(self, user_query: str, entity_top_k: int = 10, relation_top_k: int = 15) -> Dict[str, Any]:
        """Main query method that orchestrates the entire RAG pipeline"""
        
        print(f"Processing query: {user_query}")
        
        # Step 1: Extract relevant knowledge subgraph
        print("Extracting relevant knowledge subgraph...")
        knowledge_subgraph = self.extract_knowledge_subgraph(
            user_query, entity_top_k, relation_top_k
        )
        
        # Step 2: Generate context summary
        print("Generating context summary...")
        context_summary = self.generate_context_summary(knowledge_subgraph)
        
        # Step 3: Generate final answer
        print("Generating final answer...")
        answer = self.generate_answer(user_query, context_summary)
        
        return {
            "query": user_query,
            "answer": answer,
            "knowledge_subgraph": knowledge_subgraph,
            "context_summary": context_summary,
            "retrieved_entities_count": len(knowledge_subgraph["entities"]),
            "retrieved_relations_count": len(knowledge_subgraph["relations"])
        }
    
    def interactive_query_loop(self):
        """Interactive query loop for testing"""
        print("\n🔍 Graph RAG Interactive Query Interface")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nEnter your query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process query
                result = self.query(user_input)
                
                print(f"\n📊 Retrieved {result['retrieved_entities_count']} entities and {result['retrieved_relations_count']} relations")
                print(f"\n💡 Answer:")
                print(result['answer'])
                print("\n" + "="*50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")

def main():
    """Main function for testing queries"""
    
    # Initialize query engine
    query_engine = GraphRAGQueryEngine()
    
    # Example queries for testing
    test_queries = [
        "What are the main conflicts involving South Korea in 2014?",
        "What diplomatic activities did China engage in during 2014?",
        "What relationships exist between Iran and other countries?",
        "What violent events occurred involving Syria in 2014?",
        "What are the key relationships between North Korea and South Korea?"
    ]
    
    print("Running example queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Example Query {i}: {query}")
        print(f"{'='*60}")
        
        result = query_engine.query(query)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved: {result['retrieved_entities_count']} entities, {result['retrieved_relations_count']} relations")
    
    # Start interactive mode
    query_engine.interactive_query_loop()

if __name__ == "__main__":
    main()
