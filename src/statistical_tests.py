#!/usr/bin/env python3
"""
Statistical Tests on Real Temporal Features
-------------------------------------------
Tests if fake and real news show statistically different temporal patterns
Uses REAL data - no simulation
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES_DIR = "data/features"
OUTPUT_DIR = "reports/publication"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_real_temporal_features():
    """Load features extracted from real timestamps"""
    features_path = Path(FEATURES_DIR) / "temporal_features_real.csv"
    
    if not features_path.exists():
        raise FileNotFoundError(
            f"❌ Real temporal features not found at {features_path}\n"
            "   Please run: python3 src/load_twitter15_16.py"
        )
    
    df = pd.read_csv(features_path)
    print(f"📊 Loaded {len(df)} cascades with real temporal features")
    print(f"   Real news: {sum(df['label'] == 0)}")
    print(f"   Fake news: {sum(df['label'] == 1)}")
    
    return df

def compute_effect_size(fake_values, real_values):
    """Compute Cohen's d effect size"""
    fake_mean = np.mean(fake_values)
    real_mean = np.mean(real_values)
    
    # Pooled standard deviation
    n_fake = len(fake_values)
    n_real = len(real_values)
    
    pooled_std = np.sqrt(
        ((n_fake - 1) * np.var(fake_values, ddof=1) + 
         (n_real - 1) * np.var(real_values, ddof=1)) / 
        (n_fake + n_real - 2)
    )
    
    if pooled_std == 0:
        return 0
    
    cohens_d = (fake_mean - real_mean) / pooled_std
    
    return cohens_d

def test_temporal_discrimination(df):
    """
    Test which temporal features discriminate between fake and real
    at different time windows
    """
    
    print("\n" + "="*80)
    print("📊 STATISTICAL DISCRIMINATION ANALYSIS (REAL DATA)")
    print("="*80)
    
    # Identify temporal feature columns
    time_windows = [1, 3, 6, 12, 24]
    
    results = []
    
    for window in time_windows:
        print(f"\n⏰ Analyzing {window}h window...")
        
        # Find features for this window
        window_features = [col for col in df.columns if f'_{window}h' in col]
        
        for feature in window_features:
            # Get values
            fake_values = df[df['label'] == 1][feature].dropna().values
            real_values = df[df['label'] == 0][feature].dropna().values
            
            if len(fake_values) < 5 or len(real_values) < 5:
                continue
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(fake_values, real_values)
            
            # Effect size
            cohens_d = compute_effect_size(fake_values, real_values)
            
            results.append({
                'time_window': window,
                'feature': feature,
                'fake_mean': np.mean(fake_values),
                'fake_std': np.std(fake_values),
                'real_mean': np.mean(real_values),
                'real_std': np.std(real_values),
                'cohens_d': cohens_d,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_fake': len(fake_values),
                'n_real': len(real_values)
            })
    
    # Test growth features
    print(f"\n⏰ Analyzing growth features...")
    
    growth_features = [col for col in df.columns if 'velocity' in col or 'growth_rate' in col]
    
    for feature in growth_features:
        fake_values = df[df['label'] == 1][feature].dropna().values
        real_values = df[df['label'] == 0][feature].dropna().values
        
        if len(fake_values) < 5 or len(real_values) < 5:
            continue
        
        t_stat, p_value = stats.ttest_ind(fake_values, real_values)
        cohens_d = compute_effect_size(fake_values, real_values)
        
        results.append({
            'time_window': 'growth',
            'feature': feature,
            'fake_mean': np.mean(fake_values),
            'fake_std': np.std(fake_values),
            'real_mean': np.mean(real_values),
            'real_std': np.std(real_values),
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_fake': len(fake_values),
            'n_real': len(real_values)
        })
    
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    results_df.to_csv(Path(OUTPUT_DIR) / "statistical_tests_real.csv", index=False)
    
    print("\n" + "="*80)
    print("🎯 TOP DISCRIMINATIVE FEATURES (REAL DATA)")
    print("="*80)
    
    # Show top features by effect size
    significant = results_df[results_df['significant']].copy()
    significant['abs_cohens_d'] = significant['cohens_d'].abs()
    top_features = significant.nlargest(15, 'abs_cohens_d')
    
    print("\nTop 15 Most Discriminative Features:")
    print("-" * 80)
    
    for _, row in top_features.iterrows():
        direction = "Fake > Real" if row['cohens_d'] > 0 else "Real > Fake"
        print(f"\n{row['feature']} (Window: {row['time_window']})")
        print(f"   Cohen's d: {row['cohens_d']:.3f} ({interpret_cohens_d(abs(row['cohens_d']))})")
        print(f"   p-value: {row['p_value']:.6f}")
        print(f"   Direction: {direction}")
        print(f"   Fake: {row['fake_mean']:.2f} ± {row['fake_std']:.2f}")
        print(f"   Real: {row['real_mean']:.2f} ± {row['real_std']:.2f}")
    
    return results_df

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"

def test_temporal_patterns(df):
    """
    Test specific hypotheses about temporal patterns:
    - Do fake news show burst-decay?
    - Do real news show steady-growth?
    """
    
    print("\n" + "="*80)
    print("🔬 TEMPORAL PATTERN HYPOTHESIS TESTING")
    print("="*80)
    
    # Hypothesis 1: Fake news grows faster early
    print("\n📊 H1: Fake news shows higher early velocity")
    
    early_velocity = 'velocity_1to3h'
    if early_velocity in df.columns:
        fake_early_vel = df[df['label'] == 1][early_velocity].dropna()
        real_early_vel = df[df['label'] == 0][early_velocity].dropna()
        
        if len(fake_early_vel) > 0 and len(real_early_vel) > 0:
            t_stat, p_value = stats.ttest_ind(fake_early_vel, real_early_vel)
            cohens_d = compute_effect_size(fake_early_vel, real_early_vel)
            
            print(f"   Fake mean velocity (1-3h): {fake_early_vel.mean():.3f}")
            print(f"   Real mean velocity (1-3h): {real_early_vel.mean():.3f}")
            print(f"   Cohen's d: {cohens_d:.3f}")
            print(f"   p-value: {p_value:.6f}")
            print(f"   Result: {'✅ SUPPORTED' if p_value < 0.05 and cohens_d > 0 else '❌ NOT SUPPORTED'}")
    
    # Hypothesis 2: Fake news growth rate decreases over time
    print("\n📊 H2: Fake news growth rate decreases (burst-decay)")
    
    early_growth = 'growth_rate_1to3h'
    late_growth = 'growth_rate_12to24h'
    
    if early_growth in df.columns and late_growth in df.columns:
        fake_df = df[df['label'] == 1]
        
        fake_early = fake_df[early_growth].dropna()
        fake_late = fake_df[late_growth].dropna()
        
        if len(fake_early) > 0 and len(fake_late) > 0:
            # Paired t-test (same cascades, different times)
            common_idx = fake_df[[early_growth, late_growth]].dropna().index
            
            if len(common_idx) > 5:
                early_vals = fake_df.loc[common_idx, early_growth]
                late_vals = fake_df.loc[common_idx, late_growth]
                
                t_stat, p_value = stats.ttest_rel(early_vals, late_vals)
                
                print(f"   Fake early growth rate: {early_vals.mean():.3f}")
                print(f"   Fake late growth rate: {late_vals.mean():.3f}")
                print(f"   Difference: {(early_vals.mean() - late_vals.mean()):.3f}")
                print(f"   p-value: {p_value:.6f}")
                print(f"   Result: {'✅ SUPPORTED (decay pattern)' if p_value < 0.05 and early_vals.mean() > late_vals.mean() else '❌ NOT SUPPORTED'}")
    
    # Hypothesis 3: Real news shows sustained growth
    print("\n📊 H3: Real news maintains steady growth rate")
    
    if early_growth in df.columns and late_growth in df.columns:
        real_df = df[df['label'] == 0]
        
        common_idx = real_df[[early_growth, late_growth]].dropna().index
        
        if len(common_idx) > 5:
            early_vals = real_df.loc[common_idx, early_growth]
            late_vals = real_df.loc[common_idx, late_growth]
            
            t_stat, p_value = stats.ttest_rel(early_vals, late_vals)
            
            print(f"   Real early growth rate: {early_vals.mean():.3f}")
            print(f"   Real late growth rate: {late_vals.mean():.3f}")
            print(f"   Difference: {abs(early_vals.mean() - late_vals.mean()):.3f}")
            print(f"   p-value: {p_value:.6f}")
            print(f"   Result: {'✅ SUPPORTED (steady growth)' if p_value > 0.05 or abs(early_vals.mean() - late_vals.mean()) < 0.1 else '⚠️  GROWTH CHANGES'}")

def create_effect_size_visualization(results_df):
    """Visualize effect sizes across time windows"""
    
    print("\n📊 Creating effect size visualization...")
    
    # Filter significant results
    significant = results_df[results_df['significant']].copy()
    
    if len(significant) == 0:
        print("   ⚠️  No significant differences found")
        return
    
    # Group by time window
    window_results = significant[significant['time_window'] != 'growth'].copy()
    
    if len(window_results) == 0:
        print("   ⚠️  No time-windowed features found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot effect sizes by window
    windows = sorted(window_results['time_window'].unique())
    
    for window in windows:
        window_data = window_results[window_results['time_window'] == window]
        
        # Add absolute effect size
        window_data['abs_cohens_d'] = window_data['cohens_d'].abs()

        # Get top 3 features per window
        top_features = window_data.nlargest(3, 'abs_cohens_d')

        
        for _, row in top_features.iterrows():
            ax.scatter(window, abs(row['cohens_d']), s=200, alpha=0.6)
            ax.text(window, abs(row['cohens_d']), 
                   row['feature'].split('_')[0][:6], 
                   fontsize=8, ha='center')
    
    ax.set_xlabel('Time Window (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('|Cohen\'s d| (Effect Size)', fontsize=12, fontweight='bold')
    ax.set_title('Discriminative Power of Temporal Features (Real Data)', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add interpretation lines
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / 'Figure_EffectSizes_Real.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved Figure_EffectSizes_Real.png")

def main():
    """Run all statistical tests on real data"""
    
    print("="*80)
    print("🔬 STATISTICAL ANALYSIS ON REAL TEMPORAL DATA")
    print("="*80)
    
    # Load real features
    df = load_real_temporal_features()
    
    # Test discrimination
    results_df = test_temporal_discrimination(df)
    
    # Test specific patterns
    test_temporal_patterns(df)
    
    # Visualize
    create_effect_size_visualization(results_df)
    
    print("\n" + "="*80)
    print("✅ STATISTICAL ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/statistical_tests_real.csv")
    print("\n🔬 Key Insights:")
    print("   • Identified discriminative features from REAL timestamps")
    print("   • Tested temporal pattern hypotheses")
    print("   • No simulation - all results based on actual data")

if __name__ == "__main__":
    main()