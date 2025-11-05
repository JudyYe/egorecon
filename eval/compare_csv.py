# index,ga_jmse,fa_jmse,acc_norm,pampjpe
# Mean,0.037594125838950276,0.1125558887142688,0.04148259060457349,0.008990950639902925
# 001874,0.03068135492503643,0.06545887142419815,0.04476941004395485,0.011129142716526985
# 001878,0.04721531271934509,0.28786760568618774,0.05540526658296585,0.01357049960643053
# 001880,0.04456402733922005,0.12951050698757172,0.057752251625061035,0.010466267354786396
# 001881,0.06958181411027908,0.1716664880514145,0.07450330257415771,0.010856572538614273

import pandas as pd
from fire import Fire


def find_csv1_larger_than_csv2(csv_file1, csv_file2="/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/eval/hand_metrics_chunk_-1.csv", metric="ga_jmse"):
    """
    Find indices where csv_file1[metric] < csv_file2[metric]
    
    Args:
        csv_file1: Path to first CSV file
        csv_file2: Path to second CSV file
        metric: Metric column name to compare
    
    Returns:
        List of indices where csv_file1[metric] > csv_file2[metric]
    """
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Set index column as index (skip "Mean" row)
    df1 = df1.set_index("index")
    df2 = df2.set_index("index")
    
    # Remove "Mean" row if it exists
    if "Mean" in df1.index:
        df1 = df1.drop("Mean")
    if "Mean" in df2.index:
        df2 = df2.drop("Mean")
    
    # Find common indices
    common_indices = df1.index.intersection(df2.index)
    
    # Filter to only common indices
    df1_common = df1.loc[common_indices]
    df2_common = df2.loc[common_indices]
    
    # Compare metric values
    mask = df1_common[metric] > df2_common[metric]
    larger_indices = df1_common[mask].index.tolist()
    
    # Print results
    print(f"Indices where {csv_file1}[{metric}] > {csv_file2}[{metric}]:")
    print(f"Total: {len(larger_indices)} out of {len(common_indices)} common indices")
    for idx in larger_indices:
        val1 = df1_common.loc[idx, metric]
        val2 = df2_common.loc[idx, metric]
        print(f"  {idx}: {val1:.6f} > {val2:.6f} (diff: {val1 - val2:.6f})")
    
    return larger_indices

if __name__ == "__main__":
    # Example usage

    Fire(find_csv1_larger_than_csv2)
