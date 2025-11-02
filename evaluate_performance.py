import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from query_data import GraphRAGQueryEngine
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class GraphRAGEvaluator:
    """Evaluation framework for Graph RAG system"""
    
    def __init__(self, query_engine: GraphRAGQueryEngine):
        self.query_engine = query_engine
        self.evaluation_results = {}
        
    def create_evaluation_dataset(self) -> List[Dict]:
        """Create evaluation dataset with queries and expected answers"""
        
        evaluation_data = [
            {
                "query": "What conflicts involved South Korea in 2014?",
                "expected_entities": ["South Korea", "North Korea"],
                "expected_relations": ["Criticize or denounce", "Arrest, detain, or charge with legal action"],
                "ground_truth": "South Korea was involved in various conflicts and tensions, particularly with North Korea through criticism and denouncement, and internal conflicts involving arrests and legal actions.",
                "category": "conflict_analysis"
            },
            {
                "query": "What diplomatic activities did China engage in during 2014?",
                "expected_entities": ["China", "Iran", "Xi Jinping"],
                "expected_relations": ["Express intent to meet or negotiate", "Consult"],
                "ground_truth": "China engaged in diplomatic activities including expressing intent to meet and negotiate with Iran, and Xi Jinping conducted consultations with various officials.",
                "category": "diplomacy"
            },
            {
                "query": "What relationships exist between Iran and other countries?",
                "expected_entities": ["Iran", "China"],
                "expected_relations": ["Express intent to meet or negotiate", "Praise or endorse"],
                "ground_truth": "Iran had relationships involving negotiations with China and received praise or endorsement from various countries.",
                "category": "international_relations"
            },
            {
                "query": "What violent events occurred involving Syria in 2014?",
                "expected_entities": ["Syria", "Armed Rebel (Syria)", "Combatant (Al Qaeda)"],
                "expected_relations": ["Use unconventional violence"],
                "ground_truth": "Syria experienced violent events including unconventional violence used by combatants against armed rebels.",
                "category": "violence_analysis"
            },
            {
                "query": "What are the relationships between Angela Merkel and Ukraine?",
                "expected_entities": ["Angela Merkel", "Head of Government (Ukraine)", "Ukraine"],
                "expected_relations": ["Engage in negotiation"],
                "ground_truth": "Angela Merkel engaged in negotiations with the Head of Government of Ukraine.",
                "category": "political_leaders"
            },
            {
                "query": "What activities did Qatar engage in during 2014?",
                "expected_entities": ["Qatar", "Head of Government (Qatar)"],
                "expected_relations": ["Make statement"],
                "ground_truth": "Qatar's Head of Government made statements regarding Qatar during 2014.",
                "category": "government_activities"
            },
            {
                "query": "What legal actions occurred in Nigeria in 2014?",
                "expected_entities": ["Nigeria", "Citizen (Nigeria)", "Member of the Judiciary (Nigeria)"],
                "expected_relations": ["Make an appeal or request"],
                "ground_truth": "In Nigeria, citizens made appeals or requests to members of the judiciary.",
                "category": "legal_activities"
            },
            {
                "query": "What criminal activities involved Somalia and India in 2014?",
                "expected_entities": ["Somalia", "Criminal (Somalia)", "Citizen (India)", "India"],
                "expected_relations": ["Abduct, hijack, or take hostage"],
                "ground_truth": "Criminals from Somalia engaged in abduction, hijacking, or taking hostage of Indian citizens.",
                "category": "criminal_activities"
            }
        ]
        
        return evaluation_data
    
    def evaluate_entity_retrieval(self, query: str, retrieved_entities: List[Dict], 
                                expected_entities: List[str]) -> Dict[str, float]:
        """Evaluate entity retrieval performance"""
        
        retrieved_entity_names = {entity["entity"] for entity in retrieved_entities}
        expected_entity_set = set(expected_entities)
        
        # Calculate metrics
        true_positives = len(retrieved_entity_names.intersection(expected_entity_set))
        false_positives = len(retrieved_entity_names - expected_entity_set)
        false_negatives = len(expected_entity_set - retrieved_entity_names)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def evaluate_relation_retrieval(self, query: str, retrieved_relations: List[Dict], 
                                  expected_relations: List[str]) -> Dict[str, float]:
        """Evaluate relation retrieval performance"""
        
        retrieved_relation_names = {relation["relation"] for relation in retrieved_relations}
        expected_relation_set = set(expected_relations)
        
        # Calculate metrics
        true_positives = len(retrieved_relation_names.intersection(expected_relation_set))
        false_positives = len(retrieved_relation_names - expected_relation_set)
        false_negatives = len(expected_relation_set - retrieved_relation_names)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def evaluate_answer_quality(self, generated_answer: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate answer quality using embedding similarity"""
        
        # Generate embeddings for both answers
        generated_embedding = self.query_engine.embed_query(generated_answer)
        ground_truth_embedding = self.query_engine.embed_query(ground_truth)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([generated_embedding], [ground_truth_embedding])[0][0]
        
        # Simple keyword overlap metric
        generated_words = set(generated_answer.lower().split())
        ground_truth_words = set(ground_truth.lower().split())
        
        word_overlap = len(generated_words.intersection(ground_truth_words)) / len(ground_truth_words.union(generated_words))
        
        return {
            "semantic_similarity": float(similarity),
            "word_overlap": word_overlap
        }
    
    def evaluate_response_time(self, query: str) -> float:
        """Evaluate response time for a query"""
        start_time = time.time()
        self.query_engine.query(query)
        end_time = time.time()
        return end_time - start_time
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on the evaluation dataset"""
        
        print("🔍 Starting comprehensive evaluation of Graph RAG system...")
        
        evaluation_data = self.create_evaluation_dataset()
        results = {
            "entity_retrieval": [],
            "relation_retrieval": [],
            "answer_quality": [],
            "response_times": [],
            "detailed_results": []
        }
        
        for i, eval_item in enumerate(evaluation_data, 1):
            print(f"\n📊 Evaluating query {i}/{len(evaluation_data)}: {eval_item['query']}")
            
            # Get query results
            query_result = self.query_engine.query(eval_item["query"])
            
            # Evaluate entity retrieval
            entity_metrics = self.evaluate_entity_retrieval(
                eval_item["query"],
                query_result["knowledge_subgraph"]["entities"],
                eval_item["expected_entities"]
            )
            
            # Evaluate relation retrieval
            relation_metrics = self.evaluate_relation_retrieval(
                eval_item["query"],
                query_result["knowledge_subgraph"]["relations"],
                eval_item["expected_relations"]
            )
            
            # Evaluate answer quality
            answer_metrics = self.evaluate_answer_quality(
                query_result["answer"],
                eval_item["ground_truth"]
            )
            
            # Evaluate response time
            response_time = self.evaluate_response_time(eval_item["query"])
            
            # Store results
            results["entity_retrieval"].append(entity_metrics)
            results["relation_retrieval"].append(relation_metrics)
            results["answer_quality"].append(answer_metrics)
            results["response_times"].append(response_time)
            
            detailed_result = {
                "query": eval_item["query"],
                "category": eval_item["category"],
                "entity_metrics": entity_metrics,
                "relation_metrics": relation_metrics,
                "answer_metrics": answer_metrics,
                "response_time": response_time,
                "generated_answer": query_result["answer"],
                "ground_truth": eval_item["ground_truth"]
            }
            results["detailed_results"].append(detailed_result)
            
            print(f"   Entity F1: {entity_metrics['f1']:.3f}")
            print(f"   Relation F1: {relation_metrics['f1']:.3f}")
            print(f"   Semantic Similarity: {answer_metrics['semantic_similarity']:.3f}")
            print(f"   Response Time: {response_time:.3f}s")
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(results)
        results["overall_metrics"] = overall_metrics
        
        return results
    
    def calculate_overall_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        
        # Entity retrieval metrics
        entity_f1_scores = [result["f1"] for result in results["entity_retrieval"]]
        entity_precision_scores = [result["precision"] for result in results["entity_retrieval"]]
        entity_recall_scores = [result["recall"] for result in results["entity_retrieval"]]
        
        # Relation retrieval metrics
        relation_f1_scores = [result["f1"] for result in results["relation_retrieval"]]
        relation_precision_scores = [result["precision"] for result in results["relation_retrieval"]]
        relation_recall_scores = [result["recall"] for result in results["relation_retrieval"]]
        
        # Answer quality metrics
        semantic_similarities = [result["semantic_similarity"] for result in results["answer_quality"]]
        word_overlaps = [result["word_overlap"] for result in results["answer_quality"]]
        
        # Response times
        response_times = results["response_times"]
        
        return {
            "entity_f1_mean": np.mean(entity_f1_scores),
            "entity_f1_std": np.std(entity_f1_scores),
            "entity_precision_mean": np.mean(entity_precision_scores),
            "entity_recall_mean": np.mean(entity_recall_scores),
            
            "relation_f1_mean": np.mean(relation_f1_scores),
            "relation_f1_std": np.std(relation_f1_scores),
            "relation_precision_mean": np.mean(relation_precision_scores),
            "relation_recall_mean": np.mean(relation_recall_scores),
            
            "semantic_similarity_mean": np.mean(semantic_similarities),
            "semantic_similarity_std": np.std(semantic_similarities),
            "word_overlap_mean": np.mean(word_overlaps),
            
            "response_time_mean": np.mean(response_times),
            "response_time_std": np.std(response_times),
            "response_time_95th_percentile": np.percentile(response_times, 95)
        }
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate a comprehensive evaluation report"""
        
        overall = results["overall_metrics"]
        
        report = f"""
# Graph RAG System Evaluation Report

## Overall Performance Summary

### Entity Retrieval Performance
- **Average F1 Score**: {overall['entity_f1_mean']:.3f} (±{overall['entity_f1_std']:.3f})
- **Average Precision**: {overall['entity_precision_mean']:.3f}
- **Average Recall**: {overall['entity_recall_mean']:.3f}

### Relation Retrieval Performance
- **Average F1 Score**: {overall['relation_f1_mean']:.3f} (±{overall['relation_f1_std']:.3f})
- **Average Precision**: {overall['relation_precision_mean']:.3f}
- **Average Recall**: {overall['relation_recall_mean']:.3f}

### Answer Quality
- **Semantic Similarity**: {overall['semantic_similarity_mean']:.3f} (±{overall['semantic_similarity_std']:.3f})
- **Word Overlap**: {overall['word_overlap_mean']:.3f}

### Performance Metrics
- **Average Response Time**: {overall['response_time_mean']:.3f}s (±{overall['response_time_std']:.3f}s)
- **95th Percentile Response Time**: {overall['response_time_95th_percentile']:.3f}s

## Detailed Results by Category

"""
        
        # Group results by category
        category_results = {}
        for result in results["detailed_results"]:
            category = result["category"]
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        for category, category_items in category_results.items():
            report += f"\n### {category.replace('_', ' ').title()}\n"
            
            avg_entity_f1 = np.mean([item["entity_metrics"]["f1"] for item in category_items])
            avg_relation_f1 = np.mean([item["relation_metrics"]["f1"] for item in category_items])
            avg_semantic_sim = np.mean([item["answer_metrics"]["semantic_similarity"] for item in category_items])
            
            report += f"- Entity F1: {avg_entity_f1:.3f}\n"
            report += f"- Relation F1: {avg_relation_f1:.3f}\n"
            report += f"- Semantic Similarity: {avg_semantic_sim:.3f}\n"
        
        report += "\n## Recommendations\n"
        
        if overall['entity_f1_mean'] < 0.7:
            report += "- **Entity Retrieval**: Consider improving entity embedding quality or increasing retrieval parameters.\n"
        
        if overall['relation_f1_mean'] < 0.7:
            report += "- **Relation Retrieval**: Consider enhancing relation context generation or adjusting similarity thresholds.\n"
        
        if overall['semantic_similarity_mean'] < 0.7:
            report += "- **Answer Quality**: Consider improving the LLM prompting strategy or context summarization.\n"
        
        if overall['response_time_mean'] > 5.0:
            report += "- **Performance**: Consider optimizing Milvus indexing or reducing embedding dimensions for faster retrieval.\n"
        
        return report
    
    def create_visualizations(self, results: Dict):
        """Create visualization plots for evaluation results"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Graph RAG System Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. F1 Scores Comparison
        categories = [result["category"] for result in results["detailed_results"]]
        entity_f1s = [result["entity_metrics"]["f1"] for result in results["detailed_results"]]
        relation_f1s = [result["relation_metrics"]["f1"] for result in results["detailed_results"]]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, entity_f1s, width, label='Entity F1', alpha=0.8)
        axes[0, 0].bar(x + width/2, relation_f1s, width, label='Relation F1', alpha=0.8)
        axes[0, 0].set_xlabel('Query Category')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('Entity vs Relation Retrieval F1 Scores')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([cat.replace('_', '\n') for cat in categories], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Answer Quality Metrics
        semantic_sims = [result["answer_metrics"]["semantic_similarity"] for result in results["detailed_results"]]
        word_overlaps = [result["answer_metrics"]["word_overlap"] for result in results["detailed_results"]]
        
        axes[0, 1].scatter(semantic_sims, word_overlaps, alpha=0.7, s=100)
        axes[0, 1].set_xlabel('Semantic Similarity')
        axes[0, 1].set_ylabel('Word Overlap')
        axes[0, 1].set_title('Answer Quality: Semantic Similarity vs Word Overlap')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(semantic_sims, word_overlaps, 1)
        p = np.poly1d(z)
        axes[0, 1].plot(semantic_sims, p(semantic_sims), "r--", alpha=0.8)
        
        # 3. Response Time Distribution
        response_times = results["response_times"]
        axes[1, 0].hist(response_times, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(response_times), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(response_times):.2f}s')
        axes[1, 0].set_xlabel('Response Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Response Time Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Overall Performance Radar Chart
        metrics = ['Entity F1', 'Relation F1', 'Semantic Sim', 'Word Overlap', 'Speed Score']
        values = [
            results["overall_metrics"]["entity_f1_mean"],
            results["overall_metrics"]["relation_f1_mean"],
            results["overall_metrics"]["semantic_similarity_mean"],
            results["overall_metrics"]["word_overlap_mean"],
            1.0 - min(results["overall_metrics"]["response_time_mean"] / 10.0, 1.0)  # Speed score (inverted)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        axes[1, 1].plot(angles, values, 'o-', linewidth=2)
        axes[1, 1].fill(angles, values, alpha=0.25)
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Overall Performance Radar')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('graph_rag_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Evaluation visualizations saved as 'graph_rag_evaluation_results.png'")

def main():
    """Main evaluation function"""
    
    print("🚀 Initializing Graph RAG Evaluation...")
    
    # Initialize query engine
    query_engine = GraphRAGQueryEngine()
    
    # Initialize evaluator
    evaluator = GraphRAGEvaluator(query_engine)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    # Generate and save report
    report = evaluator.generate_evaluation_report(results)
    with open('evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("📋 EVALUATION REPORT")
    print("="*60)
    print(report)
    
    # Create visualizations
    evaluator.create_visualizations(results)
    
    # Save detailed results
    with open('detailed_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Evaluation completed successfully!")
    print("📁 Files saved:")
    print("   - evaluation_report.md")
    print("   - detailed_evaluation_results.json")
    print("   - graph_rag_evaluation_results.png")

if __name__ == "__main__":
    main()
