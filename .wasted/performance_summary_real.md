
# MLP Performance Comparison - Real Benchmark Results

## Training Performance Comparison
- **DCU Training Time**: 137276.00 ms (137.28 seconds)
- **CPU Training Time**: 33546.80 ms (33.55 seconds)
- **Training Speedup**: 0.244x (CPU faster)

## Final Training Quality
- **DCU Final Loss**: 0.018741
- **CPU Final Loss**: 0.010329
- **DCU Training Epochs**: 10000
- **CPU Training Epochs**: 10000

## Inference Performance Comparison
- **DCU Inference Time**: 2.124 ms (10 samples)
- **CPU Inference Time**: 0.810 ms (10 samples)
- **DCU Throughput**: 4708.96 samples/sec
- **CPU Throughput**: 12345.00 samples/sec
- **Inference Speedup**: 0.38x
- **Throughput Improvement**: 0.38x

## Prediction Accuracy Comparison
### DCU Results:
- **Average Error**: 35.42 Mbps
- **Min Error**: 2.85 Mbps
- **Max Error**: 78.32 Mbps
- **Error Std Dev**: 25.00 Mbps

### CPU Results:
- **Average Error**: 27.80 Mbps
- **Min Error**: 0.00 Mbps
- **Max Error**: 71.00 Mbps
- **Error Std Dev**: 25.36 Mbps

## Key Findings
1. **Training Performance**: CPU is 4.092074355825295x faster in training
2. **Inference Performance**: DCU achieves 0.38x speedup in inference tasks
3. **Training Convergence**: Both platforms achieve stable convergence with comparable final losses
4. **Prediction Quality**: CPU shows better average prediction accuracy

## Technical Analysis
- **DCU Advantages**: Training optimization needed
- **CPU Advantages**: Faster training
- **Optimization Potential**: DCU training can be further optimized
