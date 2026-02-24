import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os

def read_rmsd_file(filename):
    """
    读取RMSD数据文件，使用行号作为索引
    
    Parameters:
    -----------
    filename : str
        文件名，每行一个RMSD值
    
    Returns:
    --------
    rmsd_values : numpy.ndarray
        RMSD值数组
    frame_indices : numpy.ndarray
        帧索引数组（从0开始的行号）
    """
    rmsd_values = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                # 提取每行的第一个数值
                value = float(line.split()[0])
                rmsd_values.append(value)
            except ValueError:
                print(f"警告: 第 {line_num} 行无法解析: {line}")
                continue
    
    rmsd_values = np.array(rmsd_values)
    frame_indices = np.arange(len(rmsd_values))
    
    return rmsd_values, frame_indices

def cluster_rmsd_1d(rmsd_values, frame_indices, method='kmeans', n_clusters=3, threshold=0.1):
    """
    对一维RMSD值进行聚类
    """
    n_frames = len(rmsd_values)
    
    # 根据方法进行聚类
    if method == 'threshold':
        sorted_idx = np.argsort(rmsd_values)
        labels = np.zeros(n_frames, dtype=int)
        current_label = 0
        labels[sorted_idx[0]] = current_label
        
        for i in range(1, n_frames):
            if rmsd_values[sorted_idx[i]] - rmsd_values[sorted_idx[i-1]] > threshold:
                current_label += 1
            labels[sorted_idx[i]] = current_label
            
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(rmsd_values.reshape(-1, 1))
        
    elif method == 'hierarchical':
        Z = linkage(rmsd_values.reshape(-1, 1), method='ward')
        if n_clusters:
            labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        else:
            labels = fcluster(Z, threshold, criterion='distance') - 1
            
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=threshold, min_samples=2)
        labels = dbscan.fit_predict(rmsd_values.reshape(-1, 1))
        unique_labels = np.unique(labels)
        label_map = {old: i for i, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        
    else:
        raise ValueError(f"不支持的聚类方法: {method}")
    
    # 整理聚类结果
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        line_number = frame_indices[idx]
        clusters[label].append(line_number)
    
    # 找出每一类的代表性行号
    representatives = {}
    stats = {}
    
    for label, indices_list in clusters.items():
        cluster_mask = labels == label
        cluster_values = rmsd_values[cluster_mask]
        cluster_indices = frame_indices[cluster_mask]
        
        stats[label] = {
            'mean': np.mean(cluster_values),
            'std': np.std(cluster_values),
            'min': np.min(cluster_values),
            'max': np.max(cluster_values),
            'size': len(indices_list),
            'rmsd_values': cluster_values.tolist(),
            'indices': indices_list
        }
        
        mean_value = stats[label]['mean']
        closest_idx_local = np.argmin(np.abs(cluster_values - mean_value))
        representatives[label] = cluster_indices[closest_idx_local]
    
    return clusters, representatives, stats, labels

def save_clustering_results(clusters, representatives, stats, rmsd_values, frame_indices, 
                           output_prefix):
    """
    保存聚类结果到文件，使用行号作为索引
    """
    # 保存每个簇的详细信息
    with open(f"{output_prefix}_details.txt", 'w') as f:
        f.write("#" + "="*80 + "\n")
        f.write("# RMSD Clustering Results\n")
        f.write("#" + "="*80 + "\n\n")
        f.write(f"Total frames: {len(rmsd_values)}\n")
        f.write(f"Clustering method: {output_prefix.split('_')[-3]}\n")
        f.write(f"Number of clusters: {len(clusters)}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("#" + "="*80 + "\n\n")
        
        for label in sorted(clusters.keys()):
            line_numbers = sorted(clusters[label])
            rep_line = representatives[label]
            rep_rmsd = rmsd_values[rep_line]
            s = stats[label]
            
            f.write(f"CLUSTER {label}:\n")
            f.write(f"{'='*40}\n")
            f.write(f"  Size: {len(line_numbers)} frames\n")
            f.write(f"  Representative line: {rep_line} (RMSD: {rep_rmsd:.6f})\n")
            f.write(f"  Statistics:\n")
            f.write(f"    - Mean RMSD: {s['mean']:.6f}\n")
            f.write(f"    - Std Dev: {s['std']:.6f}\n")
            f.write(f"    - Min RMSD: {s['min']:.6f}\n")
            f.write(f"    - Max RMSD: {s['max']:.6f}\n")
            f.write(f"  Line numbers: {line_numbers}\n\n")
    
    # 保存代表性行号列表
    with open(f"{output_prefix}_representatives.txt", 'w') as f:
        f.write("#" + "="*60 + "\n")
        f.write("# Representative lines for each cluster\n")
        f.write("#" + "="*60 + "\n")
        f.write("# Cluster_ID\tLine_Number\tRMSD_Value\n")
        f.write("#" + "-"*60 + "\n")
        for label in sorted(representatives.keys()):
            rep_line = representatives[label]
            f.write(f"{label}\t{rep_line}\t{rmsd_values[rep_line]:.6f}\n")
    
    # 保存所有行的聚类标签
    with open(f"{output_prefix}_labels.txt", 'w') as f:
        f.write("#" + "="*60 + "\n")
        f.write("# Frame clustering labels\n")
        f.write("#" + "="*60 + "\n")
        f.write("# Line_Number\tRMSD_Value\tCluster_ID\n")
        f.write("#" + "-"*60 + "\n")
        
        labels_dict = {}
        for label, lines in clusters.items():
            for line_num in lines:
                labels_dict[line_num] = label
        
        for line_num in range(len(rmsd_values)):
            label = labels_dict.get(line_num, -1)
            f.write(f"{line_num}\t{rmsd_values[line_num]:.6f}\t{label}\n")
    
    # 保存每个簇的RMSD值列表
    with open(f"{output_prefix}_cluster_values.txt", 'w') as f:
        f.write("#" + "="*80 + "\n")
        f.write("# Cluster RMSD values by line number\n")
        f.write("#" + "="*80 + "\n\n")
        
        for label in sorted(clusters.keys()):
            line_numbers = sorted(clusters[label])
            cluster_rmsds = [rmsd_values[ln] for ln in line_numbers]
            
            f.write(f"CLUSTER {label} (n={len(line_numbers)}):\n")
            f.write(f"{'-'*40}\n")
            f.write(f"  RMSD values: {[f'{x:.6f}' for x in cluster_rmsds]}\n")
            f.write(f"  Line numbers: {line_numbers}\n\n")
    
    # 保存统计摘要
    with open(f"{output_prefix}_summary.txt", 'w') as f:
        f.write("#" + "="*80 + "\n")
        f.write("# RMSD Clustering Summary\n")
        f.write("#" + "="*80 + "\n\n")
        
        f.write(f"Total frames: {len(rmsd_values)}\n")
        f.write(f"Clustering method: {output_prefix.split('_')[-3]}\n")
        f.write(f"Number of clusters: {len(clusters)}\n\n")
        
        f.write("Cluster Statistics:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Cluster':<10} {'Size':<8} {'Mean RMSD':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Rep Line':<10}\n")
        f.write("-"*60 + "\n")
        
        for label in sorted(stats.keys()):
            s = stats[label]
            rep_line = representatives[label]
            f.write(f"{label:<10} {s['size']:<8} {s['mean']:<12.6f} {s['std']:<10.6f} "
                   f"{s['min']:<10.6f} {s['max']:<10.6f} {rep_line:<10}\n")
    
    print(f"\n文本结果已保存到:")
    print(f"  ├─ {output_prefix}_details.txt")
    print(f"  ├─ {output_prefix}_representatives.txt")
    print(f"  ├─ {output_prefix}_labels.txt")
    print(f"  ├─ {output_prefix}_cluster_values.txt")
    print(f"  └─ {output_prefix}_summary.txt")

def suggest_n_clusters(rmsd_values, max_clusters=10):
    """
    使用肘部法则建议聚类数量
    """
    from sklearn.cluster import KMeans
    import warnings
    warnings.filterwarnings('ignore')
    
    if len(rmsd_values) < 3:
        return 2
    
    inertias = []
    K_range = range(2, min(max_clusters, len(rmsd_values)//2 + 1))
    
    if len(K_range) == 0:
        return 2
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rmsd_values.reshape(-1, 1))
        inertias.append(kmeans.inertia_)
    
    if len(inertias) > 1:
        diffs = np.diff(inertias)
        if len(diffs) > 1:
            diff_rates = diffs[:-1] / diffs[1:]
            
            for i, rate in enumerate(diff_rates):
                if rate > 2.0:
                    suggested_k = i + 2
                    break
            else:
                suggested_k = K_range[np.argmin(diffs)] if len(diffs) > 0 else 3
        else:
            suggested_k = K_range[0]
    else:
        suggested_k = 2
    
    return suggested_k

def main(rmsd_file, method='kmeans', n_clusters=None, threshold=None):
    """
    主函数
    """
    print("="*80)
    print("RMSD聚类分析")
    print("="*80)
    
    # 读取数据
    print(f"读取文件: {rmsd_file}")
    rmsd_values, frame_indices = read_rmsd_file(rmsd_file)
    print(f"成功读取 {len(rmsd_values)} 个RMSD值")
    print(f"行号范围: [{frame_indices[0]}, {frame_indices[-1]}]")
    print(f"RMSD范围: [{np.min(rmsd_values):.4f}, {np.max(rmsd_values):.4f}]")
    print(f"平均RMSD: {np.mean(rmsd_values):.4f} ± {np.std(rmsd_values):.4f}")
    
    # 自动选择参数
    if n_clusters is None and threshold is None:
        if method in ['kmeans', 'hierarchical']:
            n_clusters = suggest_n_clusters(rmsd_values)
            print(f"自动选择聚类数: {n_clusters}")
        elif method == 'threshold':
            threshold = np.std(rmsd_values) * 0.3
            print(f"自动选择阈值: {threshold:.4f}")
        elif method == 'dbscan':
            threshold = np.std(rmsd_values) * 0.2
            print(f"自动选择DBSCAN eps: {threshold:.4f}")
    
    # 执行聚类
    print(f"\n使用 {method.upper()} 方法进行聚类...")
    clusters, representatives, stats, labels = cluster_rmsd_1d(
        rmsd_values, 
        frame_indices,
        method=method,
        n_clusters=n_clusters,
        threshold=threshold
    )
    
    # 打印结果
    print(f"\n聚类完成! 共 {len(clusters)} 个簇")
    print("\n" + "="*80)
    print("聚类结果详情:")
    print("="*80)
    
    for label in sorted(clusters.keys()):
        s = stats[label]
        rep_line = representatives[label]
        print(f"\n簇 {label}:")
        print(f"  ├─ 大小: {s['size']} 帧")
        print(f"  ├─ RMSD范围: [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"  ├─ 平均值: {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"  └─ 代表性行号: {rep_line} (RMSD: {rmsd_values[rep_line]:.4f})")
    
    # 生成输出文件名前缀
    base_name = os.path.splitext(os.path.basename(rmsd_file))[0]
    output_prefix = f"{base_name}_clusters_{method}"
    if method in ['kmeans', 'hierarchical'] and n_clusters:
        output_prefix += f"_k{n_clusters}"
    elif method in ['threshold', 'dbscan'] and threshold:
        output_prefix += f"_t{threshold:.3f}"
    
    # 保存文本结果
    save_clustering_results(clusters, representatives, stats, rmsd_values, frame_indices, output_prefix)
    
    return clusters, representatives, stats

# 命令行接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='对一维RMSD值进行聚类分析，使用行号作为索引')
    parser.add_argument('input_file', help='RMSD数据文件（每行一个RMSD值）')
    parser.add_argument('-m', '--method', default='kmeans', 
                       choices=['kmeans', 'hierarchical', 'dbscan', 'threshold'],
                       help='聚类方法 (默认: kmeans)')
    parser.add_argument('-k', '--n_clusters', type=int, 
                       help='聚类数量（用于kmeans和hierarchical）')
    parser.add_argument('-t', '--threshold', type=float, 
                       help='聚类阈值（用于threshold和dbscan）')
    
    args = parser.parse_args()
    
    # 运行主程序
    clusters, reps, stats = main(
        args.input_file,
        method=args.method,
        n_clusters=args.n_clusters,
        threshold=args.threshold
    )
