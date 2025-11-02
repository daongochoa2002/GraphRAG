"""
Configuration file for Graph RAG application
Modify these settings to customize the behavior of the system
"""

# LLM Configuration - Using Free Models
LLM_CONFIG = {
    "model_name": "gpt2",  # Free model
    "model_type": "transformers",
    "temperature": 0.3,
    "max_tokens": 400,
    "device": "cpu"
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "batch_size": 32
}

# Milvus/Zilliz Cloud Configuration
MILVUS_CONFIG = {
    # Local Milvus settings
    "host": "localhost",
    "port": "19530",
    
    # Zilliz Cloud settings (will override local if provided)
    "zilliz_uri": None,  # Set in .env as ZILLIZ_CLOUD_URI
    "zilliz_api_key": None,  # Set in .env as ZILLIZ_API_KEY
    
    # Index settings
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "nlist": 1024
}

# Collection Names
COLLECTIONS = {
    "entities": "graph_rag_entities",
    "relations": "graph_rag_relations"
}

# Query Configuration
QUERY_CONFIG = {
    "entity_top_k": 10,
    "relation_top_k": 15,
    "max_context_length": 4000,
    "similarity_threshold": 0.5
}

# Data Processing Configuration
DATA_CONFIG = {
    "max_entities_for_llm_context": 30,  # Reduced for free model efficiency
    "batch_size": 25,
    "max_triplets_per_batch": 500
}

# Evaluation Configuration
EVAL_CONFIG = {
    "test_queries_limit": 10,
    "evaluation_metrics": [
        "precision", "recall", "f1", 
        "semantic_similarity", "word_overlap", 
        "response_time"
    ],
    "visualization_dpi": 300
}

# Prompts for LLM Context Generation
PROMPTS = {
    "entity_context": """
Given the entity "{entity}" and its related knowledge graph triplets:
{related_triplets}

Please provide a comprehensive description of this entity including:
1. What type of entity this is (person, organization, country, etc.)
2. Key characteristics and attributes
3. Important relationships and roles
4. Context within the domain

Keep the description concise but informative (2-3 sentences).
""",
    
    "relation_context": """
Given the relation "{relation}" and example triplets where it appears:
{example_triplets}

Please provide a clear description of this relation including:
1. What this relation means
2. The typical types of entities it connects
3. The semantic meaning and implications
4. Context within the domain

Keep the description concise but informative (1-2 sentences).
""",
    
    "answer_generation_system": """You are an expert analyst with access to a knowledge graph containing geopolitical events and relationships. Your task is to provide comprehensive, accurate answers based on the retrieved knowledge graph information.

Guidelines:
1. Use the provided entities and relationships to construct your answer
2. Be specific and cite relevant information from the context
3. If the information is incomplete, acknowledge limitations
4. Provide insights and connections between different pieces of information
5. Structure your response clearly and logically
""",
    
    "answer_generation_user": """
Query: {query}

Retrieved Knowledge Graph Context:
{context_summary}

Please provide a comprehensive answer to the query based on the retrieved knowledge graph information. Make sure to:
1. Directly address the query
2. Use specific information from the entities and relationships provided
3. Explain any relevant connections or patterns
4. Provide context about timing if temporal information is available
5. Acknowledge if the available information is limited for certain aspects of the query
"""
}

# File Paths
PATHS = {
    "dataset": "datasets/icews14/train.txt",
    "entity_contexts": "entity_contexts.json",
    "relation_contexts": "relation_contexts.json",
    "evaluation_report": "evaluation_report.md",
    "detailed_results": "detailed_evaluation_results.json",
    "visualization": "graph_rag_evaluation_results.png"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "graph_rag.log"
}
