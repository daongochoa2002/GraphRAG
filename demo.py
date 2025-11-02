#!/usr/bin/env python3
"""
Demo script for Graph RAG application
This script provides a quick demonstration of the system's capabilities
"""

import os
import sys
import time
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                     Graph RAG Demo                          ║
║             Retrieval Augmented Generation                  ║
║              with Knowledge Graph Data                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_system_ready():
    """Quick check if system is ready"""
    print("🔍 Checking system status...")
    
    # Check if required files exist
    required_files = [
        "upload_data.py", 
        "query_data.py", 
        "evaluate_performance.py",
        ".env"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            return False
    
    # Check if dataset exists
    if not os.path.exists("datasets/icews14/train.txt"):
        print("❌ Dataset not found at datasets/icews14/train.txt")
        return False
    
    print("✅ System files ready")
    return True

def demo_data_upload():
    """Demonstrate data upload process"""
    print("\n" + "="*60)
    print("📊 DEMO: Data Upload and Processing")
    print("="*60)
    
    print("""
This step will:
1. Load knowledge graph data from ICEWS14 dataset
2. Extract entities and relations
3. Generate rich context using LLM
4. Create embeddings using sentence transformers
5. Store everything in Milvus vector database

Note: This requires OpenAI API key and running Milvus instance.
""")
    
    response = input("Do you want to run data upload? (y/n): ").lower().strip()
    
    if response == 'y':
        print("🚀 Starting data upload...")
        try:
            from upload_data import main as upload_main
            upload_main()
            print("✅ Data upload completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Data upload failed: {e}")
            return False
    else:
        print("⏭️  Skipping data upload")
        return False

def demo_queries():
    """Demonstrate query functionality"""
    print("\n" + "="*60)
    print("🔍 DEMO: Query and Retrieval")
    print("="*60)
    
    # Sample queries to demonstrate
    sample_queries = [
        "What conflicts involved South Korea in 2014?",
        "What diplomatic activities did China engage in during 2014?",
        "What relationships exist between Iran and other countries?",
        "What are the key relationships between North Korea and South Korea?"
    ]
    
    print("Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"{i}. {query}")
    
    print("\nThis will demonstrate the complete RAG pipeline:")
    print("- Vector similarity search for relevant entities and relations")
    print("- Knowledge subgraph extraction")
    print("- Context summarization")
    print("- LLM-based answer generation")
    
    response = input("\nDo you want to run query demo? (y/n): ").lower().strip()
    
    if response == 'y':
        try:
            from query_data import GraphRAGQueryEngine
            
            print("🚀 Initializing query engine...")
            query_engine = GraphRAGQueryEngine()
            
            print("\n🔍 Running sample queries...")
            for i, query in enumerate(sample_queries[:2], 1):  # Run first 2 queries
                print(f"\n📋 Query {i}: {query}")
                print("-" * 50)
                
                start_time = time.time()
                result = query_engine.query(query)
                end_time = time.time()
                
                print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
                print(f"📊 Retrieved: {result['retrieved_entities_count']} entities, {result['retrieved_relations_count']} relations")
                print(f"\n💡 Answer:")
                print(result['answer'])
                print("\n" + "="*50)
            
            print("✅ Query demo completed!")
            return True
            
        except Exception as e:
            print(f"❌ Query demo failed: {e}")
            print("Make sure you have uploaded data first and Milvus is running.")
            return False
    else:
        print("⏭️  Skipping query demo")
        return False

def demo_evaluation():
    """Demonstrate evaluation functionality"""
    print("\n" + "="*60)
    print("📈 DEMO: Performance Evaluation")
    print("="*60)
    
    print("""
This will run comprehensive evaluation including:
- Entity retrieval precision, recall, and F1 scores
- Relation retrieval precision, recall, and F1 scores
- Answer quality using semantic similarity
- Response time analysis
- Generate evaluation report and visualizations
""")
    
    response = input("Do you want to run evaluation demo? (y/n): ").lower().strip()
    
    if response == 'y':
        print("🚀 Starting evaluation...")
        try:
            from evaluate_performance import main as eval_main
            eval_main()
            print("✅ Evaluation completed successfully!")
            print("📁 Check these generated files:")
            print("   - evaluation_report.md")
            print("   - detailed_evaluation_results.json")
            print("   - graph_rag_evaluation_results.png")
            return True
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return False
    else:
        print("⏭️  Skipping evaluation demo")
        return False

def demo_interactive_mode():
    """Demonstrate interactive query mode"""
    print("\n" + "="*60)
    print("💬 DEMO: Interactive Query Mode")
    print("="*60)
    
    print("This will start an interactive session where you can ask custom questions.")
    
    response = input("Do you want to start interactive mode? (y/n): ").lower().strip()
    
    if response == 'y':
        try:
            from query_data import GraphRAGQueryEngine
            
            query_engine = GraphRAGQueryEngine()
            query_engine.interactive_query_loop()
            
        except Exception as e:
            print(f"❌ Interactive mode failed: {e}")
    else:
        print("⏭️  Skipping interactive mode")

def show_architecture():
    """Show system architecture"""
    print("\n" + "="*60)
    print("🏗️  SYSTEM ARCHITECTURE")
    print("="*60)
    
    architecture = """
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Knowledge      │    │   LLM Context    │    │   Embeddings    │
│  Graph Data     │───▶│   Generation     │───▶│   (Sentence     │
│  (ICEWS14)      │    │   (OpenAI)       │    │   Transformers) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  User Query     │    │  Knowledge       │    │    Milvus       │
│                 │───▶│  Subgraph        │◀───│   Vector DB     │
│                 │    │  Extraction      │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │  Context         │
         │              │  Summarization   │
         │              └──────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         └─────────────▶│  LLM Answer      │───▶│  Final Answer   │
                        │  Generation      │    │                 │
                        │  (OpenAI)        │    │                 │
                        └──────────────────┘    └─────────────────┘

Key Components:
• GraphDataProcessor: Processes knowledge graph data and generates contexts
• MilvusManager: Handles vector database operations
• GraphRAGQueryEngine: Implements the complete RAG pipeline
• GraphRAGEvaluator: Comprehensive evaluation framework
"""
    print(architecture)

def main():
    """Main demo function"""
    print_banner()
    
    if not check_system_ready():
        print("❌ System not ready. Please run setup first.")
        print("Run: python health_check.py")
        return
    
    show_architecture()
    
    print("\n🎯 DEMO WORKFLOW")
    print("="*30)
    
    # Check if user wants to see data upload demo
    uploaded = demo_data_upload()
    
    # Only proceed with queries if data was uploaded or already exists
    if uploaded or input("\nDo you have data already uploaded? (y/n): ").lower().strip() == 'y':
        demo_queries()
        demo_evaluation()
        demo_interactive_mode()
    else:
        print("⚠️  Please upload data first to use query and evaluation features.")
    
    print("\n🎉 Demo completed!")
    print("\n📚 For detailed instructions, see README.md")
    print("🔧 For system health check, run: python health_check.py")
    print("⚙️  For configuration options, see config.py")

if __name__ == "__main__":
    main()
