#!/usr/bin/env python3
"""
Visualize Real Temporal Evolution Patterns
------------------------------------------
Create publication-quality visualizations using REAL temporal data
No simulation - shows actual propagation patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FEATURES_DIR = "data/features"
OUTPUT_DIR = "reports/publication"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")

def load_real_features():
    """Load features from real timestamps"""
    features_path = Path(FEATURES_DIR) / "temporal_features_real.csv"
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"❌ Real features not found: {features_path}\n"
            "   Run: python3 src/load_twitter15_16.py"
        )
    
    df = pd.read_csv(features_path)
    print(f"📊 Loaded {len(df)} cascades with real temporal features")
    
    return df

def plot_size_evolution():
    """Plot cascade size growth over time windows"""
    
    print("\n📊 Creating size evolution plot...")
    
    df = load_real_features()
    
    time_windows = [1, 3, 6, 12, 24]
    
    # Get size at each window
    fake_sizes = []
    real_sizes = []
    fake_stds = []
    real_stds = []
    windows_with_data = []
    
    for window in time_windows:
        col = f'size_{window}h'
        
        if col in df.columns:
            fake_vals = df[df['label'] == 1][col].dropna()
            real_vals = df[df['label'] == 0][col].dropna()
            
            if len(fake_vals) > 0 and len(real_vals) > 0:
                fake_sizes.append(fake_vals.mean())
                real_sizes.append(real_vals.mean())
                fake_stds.append(fake_vals.std())
                real_stds.append(real_vals.std())
                windows_with_data.append(window)
    
    if len(windows_with_data) == 0:
        print("   ⚠️  No size data available across time windows")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(windows_with_data, fake_sizes, 'o-', color='#e74c3c', 
            linewidth=3, markersize=10, label='Fake News')
    ax.fill_between(windows_with_data,
                    np.array(fake_sizes) - np.array(fake_stds),
                    np.array(fake_sizes) + np.array(fake_stds),
                    alpha=0.2, color='#e74c3c')
    
    ax.plot(windows_with_data, real_sizes, 's-', color='#27ae60', 
            linewidth=3, markersize=10, label='Real News')
    ax.fill_between(windows_with_data,
                    np.array(real_sizes) - np.array(real_stds),
                    np.array(real_sizes) + np.array(real_stds),
                    alpha=0.2, color='#27ae60')
    
    ax.set_xlabel('Time Window (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cascade Size', fontsize=13, fontweight='bold')
    ax.set_title('Cascade Size Evolution: Fake vs Real (Real Data)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'Figure_SizeEvolution_Real.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved Figure_SizeEvolution_Real.png")

def plot_velocity_comparison():
    """Plot growth velocity across time periods"""
    
    print("\n📊 Creating velocity comparison plot...")
    
    df = load_real_features()
    
    # Find velocity columns
    velocity_cols = [col for col in df.columns if 'velocity' in col and 'to' in col]
    
    if len(velocity_cols) == 0:
        print("   ⚠️  No velocity features found")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_positions = np.arange(len(velocity_cols))
    width = 0.35
    
    fake_means = []
    real_means = []
    fake_stds = []
    real_stds = []
    labels = []
    
    for col in velocity_cols:
        fake_vals = df[df['label'] == 1][col].dropna()
        real_vals = df[df['label'] == 0][col].dropna()
        
        if len(fake_vals) > 0 and len(real_vals) > 0:
            fake_means.append(fake_vals.mean())
            real_means.append(real_vals.mean())
            fake_stds.append(fake_vals.std())
            real_stds.append(real_vals.std())
            
            # Extract time period from column name
            period = col.replace('velocity_', '').replace('h', '')
            labels.append(period)
    
    if len(labels) == 0:
        print("   ⚠️  No velocity data available")
        return
    
    x_positions = np.arange(len(labels))
    
    ax.bar(x_positions - width/2, fake_means, width, 
           yerr=fake_stds, label='Fake News', 
           color='#e74c3c', alpha=0.8, capsize=5)
    
    ax.bar(x_positions + width/2, real_means, width, 
           yerr=real_stds, label='Real News', 
           color='#27ae60', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Time Period', fontsize=13, fontweight='bold')
    ax.set_ylabel('Growth Velocity (nodes/hour)', fontsize=13, fontweight='bold')
    ax.set_title('Growth Velocity Across Time: Fake vs Real (Real Data)', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'Figure_Velocity_Real.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved Figure_Velocity_Real.png")

def plot_depth_evolution():
    """Plot cascade depth evolution"""
    
    print("\n📊 Creating depth evolution plot...")
    
    df = load_real_features()
    
    time_windows = [1, 3, 6, 12, 24]
    
    fake_depths = []
    real_depths = []
    windows_with_data = []
    
    for window in time_windows:
        col = f'depth_{window}h'
        
        if col in df.columns:
            fake_vals = df[df['label'] == 1][col].dropna()
            real_vals = df[df['label'] == 0][col].dropna()
            
            if len(fake_vals) > 0 and len(real_vals) > 0:
                fake_depths.append(fake_vals.mean())
                real_depths.append(real_vals.mean())
                windows_with_data.append(window)
    
    if len(windows_with_data) == 0:
        print("   ⚠️  No depth data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(windows_with_data, fake_depths, 'o-', color='#e74c3c', 
            linewidth=3, markersize=10, label='Fake News')
    ax.plot(windows_with_data, real_depths, 's-', color='#27ae60', 
            linewidth=3, markersize=10, label='Real News')
    
    ax.set_xlabel('Time Window (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cascade Depth', fontsize=13, fontweight='bold')
    ax.set_title('Cascade Depth Evolution: Fake vs Real (Real Data)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'Figure_DepthEvolution_Real.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved Figure_DepthEvolution_Real.png")

def plot_comprehensive_comparison():
    """Create comprehensive 6-panel comparison"""
    
    print("\n📊 Creating comprehensive comparison...")
    
    df = load_real_features()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Temporal Pattern Analysis: Fake vs Real News (Real Data)', 
                 fontsize=16, fontweight='bold')
    
    # Panel 1: Size evolution
    time_windows = [1, 3, 6, 12, 24]
    fake_sizes = []
    real_sizes = []
    windows_with_data = []
    
    for window in time_windows:
        col = f'size_{window}h'
        if col in df.columns:
            fake_vals = df[df['label'] == 1][col].dropna()
            real_vals = df[df['label'] == 0][col].dropna()
            if len(fake_vals) > 0 and len(real_vals) > 0:
                fake_sizes.append(fake_vals.mean())
                real_sizes.append(real_vals.mean())
                windows_with_data.append(window)
    
    ax = axes[0, 0]
    if len(windows_with_data) > 0:
        ax.plot(windows_with_data, fake_sizes, 'o-', color='#e74c3c', linewidth=2, label='Fake')
        ax.plot(windows_with_data, real_sizes, 's-', color='#27ae60', linewidth=2, label='Real')
    ax.set_title('Cascade Size Evolution', fontweight='bold')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Size')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Depth evolution
    fake_depths = []
    real_depths = []
    depth_windows = []
    
    for window in time_windows:
        col = f'depth_{window}h'
        if col in df.columns:
            fake_vals = df[df['label'] == 1][col].dropna()
            real_vals = df[df['label'] == 0][col].dropna()
            if len(fake_vals) > 0 and len(real_vals) > 0:
                fake_depths.append(fake_vals.mean())
                real_depths.append(real_vals.mean())
                depth_windows.append(window)
    
    ax = axes[0, 1]
    if len(depth_windows) > 0:
        ax.plot(depth_windows, fake_depths, 'o-', color='#e74c3c', linewidth=2)
        ax.plot(depth_windows, real_depths, 's-', color='#27ae60', linewidth=2)
    ax.set_title('Depth Evolution', fontweight='bold')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Depth')
    ax.grid(alpha=0.3)
    
    # Panel 3: Velocity
    velocity_cols = [col for col in df.columns if 'velocity' in col and 'to' in col]
    
    ax = axes[0, 2]
    if len(velocity_cols) > 0:
        fake_velocities = [df[df['label'] == 1][col].dropna().mean() for col in velocity_cols]
        real_velocities = [df[df['label'] == 0][col].dropna().mean() for col in velocity_cols]
        x = np.arange(len(velocity_cols))
        ax.bar(x - 0.2, fake_velocities, 0.4, color='#e74c3c', alpha=0.8, label='Fake')
        ax.bar(x + 0.2, real_velocities, 0.4, color='#27ae60', alpha=0.8, label='Real')
        ax.set_xticks(x)
        ax.set_xticklabels([col.replace('velocity_', '').replace('h', '') for col in velocity_cols], 
                          rotation=45, ha='right', fontsize=8)
    ax.set_title('Growth Velocity', fontweight='bold')
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 4: Growth rates
    growth_cols = [col for col in df.columns if 'growth_rate' in col]
    
    ax = axes[1, 0]
    if len(growth_cols) > 0:
        fake_growth = [df[df['label'] == 1][col].dropna().mean() for col in growth_cols]
        real_growth = [df[df['label'] == 0][col].dropna().mean() for col in growth_cols]
        x = np.arange(len(growth_cols))
        ax.bar(x - 0.2, fake_growth, 0.4, color='#e74c3c', alpha=0.8)
        ax.bar(x + 0.2, real_growth, 0.4, color='#27ae60', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([col.replace('growth_rate_', '').replace('h', '') for col in growth_cols], 
                          rotation=45, ha='right', fontsize=8)
    ax.set_title('Growth Rate', fontweight='bold')
    ax.set_ylabel('Rate')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel 5: Distribution comparison
    ax = axes[1, 1]
    fake_full_size = df[df['label'] == 1]['cascade_size_full'].dropna()
    real_full_size = df[df['label'] == 0]['cascade_size_full'].dropna()
    
    if len(fake_full_size) > 0 and len(real_full_size) > 0:
        ax.hist(fake_full_size, bins=30, alpha=0.6, color='#e74c3c', label='Fake', density=True)
        ax.hist(real_full_size, bins=30, alpha=0.6, color='#27ae60', label='Real', density=True)
    ax.set_title('Final Size Distribution', fontweight='bold')
    ax.set_xlabel('Cascade Size')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 6: Time delay comparison
    ax = axes[1, 2]
    fake_delay = df[df['label'] == 1]['avg_time_delay'].dropna()
    real_delay = df[df['label'] == 0]['avg_time_delay'].dropna()
    
    if len(fake_delay) > 0 and len(real_delay) > 0:
        ax.boxplot([fake_delay, real_delay], labels=['Fake', 'Real'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax.set_title('Average Time Delay', fontweight='bold')
    ax.set_ylabel('Time Delay (minutes)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'Figure_Comprehensive_Real.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved Figure_Comprehensive_Real.png")

def main():
    """Create all visualizations"""
    
    print("="*80)
    print("📊 VISUALIZING REAL TEMPORAL PATTERNS")
    print("="*80)
    
    try:
        plot_size_evolution()
        plot_velocity_comparison()
        plot_depth_evolution()
        plot_comprehensive_comparison()
        
        print("\n" + "="*80)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"\n📁 Figures saved to: {OUTPUT_DIR}/")
        print("\n📊 Generated:")
        print("   • Figure_SizeEvolution_Real.png")
        print("   • Figure_Velocity_Real.png")
        print("   • Figure_DepthEvolution_Real.png")
        print("   • Figure_Comprehensive_Real.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've run: python3 src/load_twitter15_16.py")

if __name__ == "__main__":
    main()