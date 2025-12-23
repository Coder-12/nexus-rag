"""
Nexus RAG - Dataset Validation for Manually Collected Corpus
Validates the 15-article interconnected ML knowledge corpus
"""

import os
from pathlib import Path
from collections import defaultdict

# Expected documents by cluster
DATASET_STRUCTURE = {
    "Core Architecture": [
        "transformer_architecture.txt",
        "attention_mechanism.txt",
        "bert_architecture.txt",
        "gpt_architecture.txt",
        "encoder_decoder_models.txt",
    ],
    "Training & Optimization": [
        "transfer_learning.txt",
        "in_context_learning.txt",
        "prompt_engineering.txt",
        "reinforcement_learning_with_human_feedback.txt",
        "fine_tuning.txt",
    ],
    "RAG & Retrieval Systems": [
        "retrieval_augmented_generation.txt",
        "embeddings.txt",
        "semantic_search.txt",
        "vector_database.txt",
    ],
    "Systems Context & Safety": [
        "large_language_models.txt",
        "ai_alignment.txt",
    ],
}

DATA_DIR = Path("data/raw")

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def validate_corpus():
    """Validate the manually collected dataset"""
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'NEXUS RAG - DATASET VALIDATION'.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")
    
    total_expected = sum(len(docs) for docs in DATASET_STRUCTURE.values())
    total_found = 0
    total_words = 0
    total_size_kb = 0
    
    cluster_stats = {}
    all_collected = []
    
    # Validate each cluster
    for cluster_name, expected_docs in DATASET_STRUCTURE.items():
        print(f"{Colors.BOLD}{cluster_name} ({len(expected_docs)} documents){Colors.END}")
        print("-" * 80)
        
        cluster_collected = []
        cluster_words = 0
        
        for doc_name in expected_docs:
            file_path = DATA_DIR / doc_name
            print(file_path)
            
            if file_path.exists():
                # Get stats
                size_bytes = file_path.stat().st_size
                size_kb = size_bytes / 1024
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    word_count = len(content.split())
                    char_count = len(content)
                
                # Quality check
                if word_count < 500:
                    quality = f"{Colors.YELLOW}⚠️  SHORT (<500 words){Colors.END}"
                elif word_count > 30000:
                    quality = f"{Colors.YELLOW}⚠️  LONG (>30K words){Colors.END}"
                else:
                    quality = f"{Colors.GREEN}✅{Colors.END}"
                
                print(f"  {quality} {doc_name:45s} {word_count:7,d} words | {size_kb:6.1f} KB")
                
                cluster_collected.append(doc_name)
                cluster_words += word_count
                total_found += 1
                total_words += word_count
                total_size_kb += size_kb
                
                all_collected.append({
                    'cluster': cluster_name,
                    'filename': doc_name,
                    'words': word_count,
                    'size_kb': size_kb
                })
            else:
                print(f"  {Colors.RED}❌ MISSING{Colors.END} {doc_name}")
        
        cluster_stats[cluster_name] = {
            'collected': len(cluster_collected),
            'expected': len(expected_docs),
            'words': cluster_words
        }
        
        print(f"  {Colors.BLUE}Cluster: {len(cluster_collected)}/{len(expected_docs)} collected | {cluster_words:,d} words{Colors.END}\n")
    
    # Overall Summary
    print(f"{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    completion_pct = (total_found / total_expected) * 100
    
    print(f"📊 Overall Progress:")
    print(f"   Collected: {Colors.GREEN}{total_found}/{total_expected}{Colors.END} documents ({completion_pct:.0f}%)")
    print(f"   Total Words: {total_words:,d}")
    print(f"   Average Words/Doc: {total_words // max(total_found, 1):,d}")
    print(f"   Total Size: {total_size_kb / 1024:.2f} MB\n")
    
    # Cluster breakdown
    print(f"📋 By Cluster:")
    for cluster_name, stats in cluster_stats.items():
        status = "✅" if stats['collected'] == stats['expected'] else "⚠️"
        print(f"   {status} {cluster_name:30s} {stats['collected']}/{stats['expected']} | {stats['words']:7,d} words")
    
    # Quality checks
    print(f"\n🔍 Quality Checks:")
    
    # Check word count distribution
    if all_collected:
        word_counts = [doc['words'] for doc in all_collected]
        min_words = min(word_counts)
        max_words = max(word_counts)
        avg_words = sum(word_counts) // len(word_counts)
        
        print(f"   Min words: {min_words:,d}")
        print(f"   Max words: {max_words:,d}")
        print(f"   Avg words: {avg_words:,d}")
        
        # Distribution check
        short_docs = [doc for doc in all_collected if doc['words'] < 3000]
        medium_docs = [doc for doc in all_collected if 3000 <= doc['words'] <= 15000]
        long_docs = [doc for doc in all_collected if doc['words'] > 15000]
        
        print(f"\n   Distribution:")
        print(f"   - Short (<3K):   {len(short_docs):2d} documents")
        print(f"   - Medium (3-15K): {len(medium_docs):2d} documents")
        print(f"   - Long (>15K):    {len(long_docs):2d} documents")
    
    # Final verdict
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    
    if total_found == total_expected:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ DATASET COMPLETE - READY FOR INGESTION{Colors.END}")
        print(f"\n{Colors.BLUE}Next Step: python scripts/create_document_loader.py{Colors.END}")
        return True
    else:
        missing = total_expected - total_found
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  INCOMPLETE: {missing} document(s) missing{Colors.END}")
        print(f"\n{Colors.BLUE}Complete collection before proceeding to Step 2{Colors.END}")
        return False

if __name__ == "__main__":
    import sys
    success = validate_corpus()
    sys.exit(0 if success else 1)