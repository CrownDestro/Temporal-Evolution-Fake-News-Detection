# Early Fake News Detection Using Temporal Propagation Features

## Overview
This project detects fake news using time-windowed temporal features extracted from real Twitter15/16 retweet cascades. We achieve 62.5% F1-score at 1 hour with only 0.2% performance cost compared to waiting 24 hours.

## Key Contributions
1. First systematic time-windowed analysis (1h-24h) with real timestamps
2. Statistical validation: Fake news shows burst-decay pattern (p<0.001)
3. Early detection: 62.5% F1 at 1h vs 62.7% at 24h (96% time savings)
4. Feature engineering improves ROC-AUC by 11.9%

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Step 1: Extract features
python3 src/load_twitter15_16.py

# Step 2: Statistical tests
python3 src/statistical_tests.py

# Step 3: Visualizations
python3 src/visualize_temporal_patterns.py

# Step 4: Train models
python3 src/improved_temporal_models.py
```

## Results
- Early Detection (1h): F1=0.625, ROC-AUC=0.692
- Full Detection (24h): F1=0.627, ROC-AUC=0.670
- Time Savings: 96% with <1% performance cost
