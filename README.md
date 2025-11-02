# Graph RAG Application

A comprehensive Graph RAG (Retrieval Augmented Generation) application that embeds entities and relations using LLMs, stores them in Milvus vector database, and provides intelligent query answering capabilities.

## 🚀 Features

- **Knowledge Graph Processing**: Extract entities and relations from temporal knowledge graph data
- **LLM-Enhanced Embeddings**: Use LLMs to generate rich context for entities and relations before embedding
- **Vector Storage**: Store embeddings in Milvus for efficient similarity search
- **Intelligent Retrieval**: Extract relevant knowledge subgraphs based on user queries
- **RAG Pipeline**: Combine retrieved knowledge with LLM generation for comprehensive answers
- **Performance Evaluation**: Comprehensive evaluation framework with multiple metrics

## 📁 Project Structure

```
GraphRAG/
├── datasets/
│   └── icews14/          # ICEWS14 temporal knowledge graph dataset
├── upload_data.py        # Data processing and upload to Milvus
├── query_data.py         # Query engine and RAG pipeline
├── evaluate_performance.py # Evaluation framework
├── requirements.txt      # Python dependencies
├── .env                 # Environment configuration
└── README.md           # This file
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Vector Database

You have two options for the vector database:

#### Option A: Zilliz Cloud (Recommended)

1. Sign up for a free Zilliz Cloud account at [cloud.zilliz.com](https://cloud.zilliz.com)
2. Create a serverless cluster
3. Get your cluster URI and API key from the dashboard
4. Update your `.env` file with the credentials (see step 3)

#### Option B: Local Milvus with Docker

```bash
# Download and start Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

### 3. Configure Environment

Edit the `.env` file with your configuration:

**For Zilliz Cloud (Recommended):**
```env
OPENAI_API_KEY=your_openai_api_key_here
ZILLIZ_CLOUD_URI=your_zilliz_cluster_uri
ZILLIZ_API_KEY=your_zilliz_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
COLLECTION_NAME=graph_rag_entities
RELATION_COLLECTION_NAME=graph_rag_relations
```

**For Local Milvus:**
```env
OPENAI_API_KEY=your_openai_api_key_here
MILVUS_HOST=localhost
MILVUS_PORT=19530
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
COLLECTION_NAME=graph_rag_entities
RELATION_COLLECTION_NAME=graph_rag_relations
```

## 🔄 Usage

### 1. Upload and Process Data

```bash
python upload_data.py
```

This script will:
- Load the ICEWS14 knowledge graph data
- Extract entities and relations
- Use LLM to generate rich context for entities and relations
- Generate embeddings using sentence transformers
- Store everything in Milvus vector database

### 2. Query the System

```bash
python query_data.py
```

This will start an interactive query interface where you can ask questions like:
- "What conflicts involved South Korea in 2014?"
- "What diplomatic activities did China engage in during 2014?"
- "What relationships exist between Iran and other countries?"

### 3. Evaluate Performance

```bash
python evaluate_performance.py
```

This will run comprehensive evaluation including:
- Entity retrieval precision, recall, and F1
- Relation retrieval precision, recall, and F1
- Answer quality using semantic similarity
- Response time analysis
- Generate evaluation report and visualizations

## 🏗️ Architecture

### Data Processing Pipeline (`upload_data.py`)

1. **Data Loading**: Load temporal knowledge graph triplets from ICEWS14 dataset
2. **Entity Context Generation**: Use LLM to generate rich descriptions for entities
3. **Relation Context Generation**: Use LLM to generate semantic descriptions for relations
4. **Embedding Generation**: Create vector embeddings using sentence transformers
5. **Vector Storage**: Store embeddings in Milvus with metadata

### Query Pipeline (`query_data.py`)

1. **Query Processing**: Convert user query to vector embedding
2. **Knowledge Retrieval**: 
   - Search for relevant entities using vector similarity
   - Search for relevant relations using vector similarity
   - Filter and combine results to create knowledge subgraph
3. **Context Generation**: Create comprehensive context summary from retrieved knowledge
4. **Answer Generation**: Use LLM with retrieved context to generate final answer

### Evaluation Framework (`evaluate_performance.py`)

1. **Entity Retrieval Evaluation**: Measure precision, recall, and F1 for entity retrieval
2. **Relation Retrieval Evaluation**: Measure precision, recall, and F1 for relation retrieval
3. **Answer Quality Evaluation**: Use semantic similarity and word overlap metrics
4. **Performance Analysis**: Measure response times and system efficiency
5. **Visualization**: Generate comprehensive charts and reports

## 📊 Dataset

The application uses the **ICEWS14** dataset, which contains:
- **Entities**: Countries, organizations, political leaders, groups
- **Relations**: Political interactions, diplomatic activities, conflicts, negotiations
- **Temporal Information**: Events timestamped for 2014
- **Scale**: 72,827 training triplets

Example triplet:
```
South Korea    Criticize or denounce    North Korea    2014-05-13
```

## 🎯 Key Components

### GraphDataProcessor
- Loads and processes knowledge graph data
- Extracts entities and relations
- Generates LLM-enhanced contexts

### MilvusManager
- Manages Milvus vector database operations
- Creates collections for entities and relations
- Handles embedding generation and storage

### GraphRAGQueryEngine
- Implements the complete RAG pipeline
- Performs similarity search and knowledge retrieval
- Generates contextual answers using LLM

### GraphRAGEvaluator
- Comprehensive evaluation framework
- Multiple evaluation metrics
- Visualization and reporting capabilities

## 📈 Performance Metrics

The evaluation framework measures:

- **Retrieval Quality**: Precision, Recall, F1 for entities and relations
- **Answer Quality**: Semantic similarity with ground truth
- **Efficiency**: Response time and throughput
- **Robustness**: Performance across different query categories

## 🔧 Customization

### Adding New Datasets
1. Implement data loader in `GraphDataProcessor`
2. Ensure triplet format: `(head_entity, relation, tail_entity, timestamp)`
3. Update evaluation dataset in `GraphRAGEvaluator`

### Modifying Embeddings
1. Change `EMBEDDING_MODEL` in `.env`
2. Update dimension in Milvus collection creation
3. Ensure model compatibility with sentence-transformers

### Enhancing LLM Integration
1. Modify prompts in context generation methods
2. Experiment with different LLM models
3. Adjust temperature and generation parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ICEWS Dataset**: Integrated Crisis Early Warning System
- **Milvus**: Vector database for similarity search
- **LangChain**: LLM application framework
- **Sentence Transformers**: Embedding models
