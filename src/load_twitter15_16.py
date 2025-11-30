#!/usr/bin/env python3
"""
Load Twitter15/Twitter16 Real Propagation Data (FIXED - Using Real Timestamps)
------------------------------------------------------------------------------
Uses actual retweet timestamps from the data
NO SIMULATION - All temporal analysis based on real time delays
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import re

# Paths
TWITTER15_DIR = "data/raw/rumor_detection_acl2017/twitter15"
TWITTER16_DIR = "data/raw/rumor_detection_acl2017/twitter16"
GRAPHS_DIR = "data/graphs_real"
FEATURES_DIR = "data/features"

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

def parse_tree_file(tree_file_path):
    """
    Parse a tree file to extract edges with REAL timestamps
    Format: ['uid', 'tweet_id', 'time_delay'] -> ['uid', 'tweet_id', 'time_delay']
    """
    edges = []
    
    with open(tree_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if '->' not in line:
                continue
            
            # Split parent and child
            parts = line.split('->')
            if len(parts) != 2:
                continue
            
            parent_str = parts[0].strip()
            child_str = parts[1].strip()
            
            # Parse node format: ['uid', 'tweet_id', 'time_delay']
            try:
                # Extract content within brackets
                parent_match = re.findall(r"'([^']*)'", parent_str)
                child_match = re.findall(r"'([^']*)'", child_str)
                
                if len(parent_match) >= 3 and len(child_match) >= 3:
                    parent_uid = parent_match[0]
                    parent_tweet = parent_match[1]
                    parent_time = float(parent_match[2]) if parent_match[2] else 0.0
                    
                    child_uid = child_match[0]
                    child_tweet = child_match[1]
                    child_time = float(child_match[2]) if child_match[2] else 0.0
                    
                    edges.append({
                        'parent_uid': parent_uid,
                        'parent_tweet': parent_tweet,
                        'parent_time': parent_time,  # REAL timestamp!
                        'child_uid': child_uid,
                        'child_tweet': child_tweet,
                        'child_time': child_time  # REAL timestamp!
                    })
            except Exception as e:
                continue
    
    return edges

def load_twitter_dataset(dataset_dir, dataset_name):
    """Load Twitter15 or Twitter16 dataset"""
    print(f"\n📂 Loading {dataset_name} dataset...")
    
    dataset_path = Path(dataset_dir)
    tree_dir = dataset_path / "tree"
    label_file = dataset_path / "label.txt"
    source_file = dataset_path / "source_tweets.txt"
    
    if not dataset_path.exists():
        print(f"   ❌ Directory not found: {dataset_path}")
        return None
    
    # Load labels
    labels = {}
    if label_file.exists():
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    label, tweet_id = parts
                    labels[tweet_id] = label.lower()
    
    print(f"   Loaded {len(labels)} labels")
    
    # Load source tweets
    source_tweets = {}
    if source_file.exists():
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    tweet_id = parts[0]
                    content = parts[1] if len(parts) > 1 else ""
                    source_tweets[tweet_id] = content
    
    print(f"   Loaded {len(source_tweets)} source tweets")
    
    # Load tree structures
    propagation_trees = {}
    
    if tree_dir.exists():
        tree_files = list(tree_dir.glob("*.txt"))
        print(f"   Found {len(tree_files)} tree files")
        
        for tree_file in tqdm(tree_files, desc=f"   Parsing {dataset_name} trees"):
            tweet_id = tree_file.stem
            edges = parse_tree_file(tree_file)
            
            if edges:
                propagation_trees[tweet_id] = edges
    
    print(f"   Parsed {len(propagation_trees)} propagation trees")
    
    return {
        'labels': labels,
        'source_tweets': source_tweets,
        'propagation_trees': propagation_trees
    }

def build_propagation_network(tweet_id, edges, label, content=""):
    """Build networkx graph from tree edges with REAL timestamps"""
    G = nx.DiGraph()
    
    # Create node mapping
    node_map = {}
    node_counter = 0
    
    # Add root node (source tweet at time 0)
    G.add_node(0, 
               node_type='source',
               tweet_id=tweet_id,
               user_id='root',
               time_delay=0.0,
               label=label,
               content=content)
    
    # Process all edges
    for edge in edges:
        parent_key = (edge['parent_uid'], edge['parent_tweet'])
        child_key = (edge['child_uid'], edge['child_tweet'])
        
        # Map parent node
        if parent_key not in node_map:
            node_counter += 1
            node_map[parent_key] = node_counter
            G.add_node(node_counter,
                      node_type='user',
                      user_id=edge['parent_uid'],
                      tweet_id=edge['parent_tweet'],
                      time_delay=edge['parent_time'])
        
        # Map child node
        if child_key not in node_map:
            node_counter += 1
            node_map[child_key] = node_counter
            G.add_node(node_counter,
                      node_type='user',
                      user_id=edge['child_uid'],
                      tweet_id=edge['child_tweet'],
                      time_delay=edge['child_time'])
        
        # Add edge
        parent_id = node_map[parent_key]
        child_id = node_map[child_key]
        
        G.add_edge(parent_id, child_id,
                  edge_type='retweet',
                  time_delay=edge['child_time'] - edge['parent_time'])
    
    # Connect root to first-level nodes
    for node in G.nodes():
        if node != 0 and G.in_degree(node) == 0:
            G.add_edge(0, node, edge_type='retweet', 
                      time_delay=G.nodes[node]['time_delay'])
    
    # Store metadata
    G.graph['tweet_id'] = tweet_id
    G.graph['label'] = 1 if label == 'false' else 0
    G.graph['num_users'] = len(G.nodes()) - 1
    G.graph['num_edges'] = len(G.edges())
    
    return G

def extract_temporal_features(G, graph_id, time_windows=[1, 3, 6, 12, 24]):
    """
    Extract REAL temporal features at different time windows
    NO SIMULATION - Uses actual time delays from data
    """
    
    # Convert time windows to minutes
    time_windows_min = [t * 60 for t in time_windows]
    
    user_nodes = [n for n in G.nodes() if n != 0]
    
    if len(user_nodes) == 0:
        return None
    
    # Get time delays for all nodes
    time_delays = [G.nodes[n].get('time_delay', 0.0) for n in user_nodes]
    
    if not time_delays or max(time_delays) == 0:
        return None
    
    # Extract features for each time window
    features_all_windows = {}
    
    for window_idx, window_minutes in enumerate(time_windows_min):
        window_hours = time_windows[window_idx]
        
        # Get nodes that appeared within this time window
        nodes_in_window = [n for n in user_nodes 
                          if G.nodes[n].get('time_delay', 0.0) <= window_minutes]
        
        if len(nodes_in_window) == 0:
            continue
        
        # Compute depths for nodes in window
        depths = {}
        for node in nodes_in_window:
            try:
                depths[node] = nx.shortest_path_length(G, 0, node)
            except:
                depths[node] = 1
        
        # Extract features for this window
        features = {
            f'size_{window_hours}h': len(nodes_in_window),
            f'depth_{window_hours}h': max(depths.values()) if depths else 1,
            f'width_{window_hours}h': sum(1 for n in nodes_in_window if depths.get(n, 0) == 1),
            f'avg_degree_{window_hours}h': np.mean([G.degree(n) for n in nodes_in_window]) if nodes_in_window else 0,
            f'max_time_{window_hours}h': max([G.nodes[n].get('time_delay', 0) for n in nodes_in_window]),
        }
        
        features_all_windows.update(features)
    
    # Compute growth metrics between windows
    growth_features = {}
    
    for i in range(1, len(time_windows)):
        prev_window = time_windows[i-1]
        curr_window = time_windows[i]
        
        prev_size_key = f'size_{prev_window}h'
        curr_size_key = f'size_{curr_window}h'
        
        if prev_size_key in features_all_windows and curr_size_key in features_all_windows:
            prev_size = features_all_windows[prev_size_key]
            curr_size = features_all_windows[curr_size_key]
            
            # Growth velocity
            time_diff = curr_window - prev_window
            velocity = (curr_size - prev_size) / time_diff if time_diff > 0 else 0
            growth_features[f'velocity_{prev_window}to{curr_window}h'] = velocity
            
            # Growth rate (normalized)
            growth_rate = (curr_size - prev_size) / (prev_size + 1)
            growth_features[f'growth_rate_{prev_window}to{curr_window}h'] = growth_rate
    
    # Combine all features
    final_features = {
        'graph_id': graph_id,
        'tweet_id': G.graph.get('tweet_id', ''),
        'label': G.graph['label'],
    }
    
    final_features.update(features_all_windows)
    final_features.update(growth_features)
    
    # Add full cascade features for comparison
    all_depths = {}
    for node in user_nodes:
        try:
            all_depths[node] = nx.shortest_path_length(G, 0, node)
        except:
            all_depths[node] = 1
    
    final_features.update({
        'cascade_size_full': len(user_nodes),
        'cascade_depth_full': max(all_depths.values()) if all_depths else 1,
        'cascade_width_full': sum(1 for n in user_nodes if all_depths.get(n, 0) == 1),
        'avg_time_delay': np.mean(time_delays),
        'max_time_delay': max(time_delays),
        'spread_velocity_full': len(user_nodes) / (max(time_delays) + 1) if time_delays else 0,
    })
    
    # Structural virality
    try:
        if len(user_nodes) > 1:
            final_features['structural_virality'] = nx.wiener_index(G.to_undirected()) / (len(user_nodes) + 1)
        else:
            final_features['structural_virality'] = 0
    except:
        final_features['structural_virality'] = 0
    
    return final_features

def process_twitter_datasets():
    """Process both Twitter15 and Twitter16 with REAL temporal data"""
    print("="*80)
    print("📥 PROCESSING REAL TWITTER DATASETS (No Simulation)")
    print("="*80)
    
    all_graphs = []
    all_features = []
    
    # Process both datasets
    for dataset_dir, dataset_name in [
        (TWITTER15_DIR, 'twitter15'),
        (TWITTER16_DIR, 'twitter16')
    ]:
        print(f"\n{'='*80}")
        dataset = load_twitter_dataset(dataset_dir, dataset_name)
        
        if dataset is None:
            print(f"⚠️  {dataset_name} not found, skipping...")
            continue
        
        labels = dataset['labels']
        source_tweets = dataset['source_tweets']
        propagation_trees = dataset['propagation_trees']
        
        # Build graphs
        print(f"\n🔨 Building propagation graphs for {dataset_name}...")
        
        processed = 0
        skipped = 0
        
        for tweet_id, edges in tqdm(propagation_trees.items(), desc=f"Processing {dataset_name}"):
            # Get label
            label = labels.get(tweet_id, 'unknown')
            
            # Convert label to binary
            if label == 'false':
                binary_label = 1  # Fake
            elif label == 'true':
                binary_label = 0  # Real
            else:
                skipped += 1
                continue
            
            # Need minimum cascade size
            if len(edges) < 2:
                skipped += 1
                continue
            
            # Get content
            content = source_tweets.get(tweet_id, "")
            
            # Build graph
            try:
                G = build_propagation_network(tweet_id, edges, label, content)
                
                # Extract temporal features (REAL timestamps!)
                features = extract_temporal_features(G, len(all_graphs))
                
                if features:
                    all_graphs.append(G)
                    all_features.append(features)
                    processed += 1
            except Exception as e:
                skipped += 1
                continue
        
        print(f"   ✅ Processed: {processed}")
        print(f"   ⏭️  Skipped: {skipped}")
    
    if not all_features:
        print("\n❌ No features extracted!")
        return None
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    print(f"\n{'='*80}")
    print(f"✅ Processed {len(all_graphs)} real propagation graphs!")
    
    # Analyze temporal coverage
    print("\n📊 Temporal Coverage Analysis:")
    
    for window in [1, 3, 6, 12, 24]:
        size_col = f'size_{window}h'
        if size_col in features_df.columns:
            has_data = features_df[size_col].notna().sum()
            print(f"   {window}h window: {has_data}/{len(features_df)} cascades ({has_data/len(features_df)*100:.1f}%)")
    
    # Compare fake vs real
    print("\n📊 Feature Comparison (Real vs Fake):")
    
    comparison_cols = ['cascade_size_full', 'cascade_depth_full', 
                      'spread_velocity_full', 'avg_time_delay']
    
    comparison = features_df.groupby('label')[comparison_cols].mean()
    comparison.index = ['Real', 'Fake']
    print(comparison.round(2))
    
    # Label distribution
    print("\n📊 Label Distribution:")
    print(features_df['label'].value_counts())
    
    # Save features
    features_path = Path(FEATURES_DIR) / "temporal_features_real.csv"
    features_df.to_csv(features_path, index=False)
    print(f"\n💾 Saved features to {features_path}")
    
    # Save sample graphs
    print("\n💾 Saving sample graphs...")
    sample_indices = np.random.choice(len(all_graphs), min(20, len(all_graphs)), replace=False)
    
    for idx in sample_indices:
        G = all_graphs[idx]
        label = 'fake' if G.graph['label'] == 1 else 'real'
        graph_path = Path(GRAPHS_DIR) / f"graph_{idx}_{label}.json"
        
        graph_data = nx.node_link_data(G)
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    print(f"   Saved {len(sample_indices)} sample graphs to {GRAPHS_DIR}/")
    
    return features_df, all_graphs

if __name__ == "__main__":
    print("🔧 Using REAL Twitter15/Twitter16 temporal data")
    print("   Dataset: Liu et al. (2018) - ACL 2017")
    print("   Timestamps: Real time delays from retweet cascades")
    
    result = process_twitter_datasets()
    
    if result:
        features_df, graphs = result
        print("\n" + "="*80)
        print("✅ REAL DATA PROCESSING COMPLETE!")
        print("="*80)
        print("\n📊 Dataset Statistics:")
        print(f"   • Total graphs: {len(graphs)}")
        print(f"   • Real news: {sum(features_df['label'] == 0)}")
        print(f"   • Fake news: {sum(features_df['label'] == 1)}")
        print(f"   • Avg cascade size: {features_df['cascade_size_full'].mean():.1f}")
        print(f"   • Avg time delay: {features_df['avg_time_delay'].mean():.1f} min")
        print("\n🚀 Next steps:")
        print("1. python3 src/statistical_tests.py")
        print("2. python3 src/visualize_temporal_patterns.py")
        print("3. python3 src/temporal_evolution_analysis.py")
    else:
        print("\n❌ Processing failed - check error messages above")