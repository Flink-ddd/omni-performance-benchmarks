import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_vram_plot(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"物理加载失败: {e}")
        return

    df.columns = [c.strip() for c in df.columns]
    
    if 'index' not in df.columns:
        print("检测到旧版 CSV，物理尝试按 2 块 GPU 进行交错切片...")
        df['index'] = [i % 2 for i in range(len(df))]
    
    df['memory.used [MiB]'] = df['memory.used [MiB]'].str.replace(' MiB', '').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    start_time = df['timestamp'].min()
    df['time_sec'] = (df['timestamp'] - start_time).dt.total_seconds()

    plt.figure(figsize=(12, 6))
    colors = ['#27ae60', '#e74c3c', '#3498db', '#f1c40f']
    
    for i, gpu_idx in enumerate(df['index'].unique()):
        gpu_data = df[df['index'] == gpu_idx].copy()
        label = f"GPU {gpu_idx}"
        
        if gpu_idx == 0:
            label += " (LLM Thinker)"
        elif gpu_idx == 1:
            label += " (Talker + Diffusion)"
            
        plt.plot(gpu_data['time_sec'], gpu_data['memory.used [MiB]'], 
                 label=label, color=colors[i % len(colors)], linewidth=2.5)

    plt.title('vLLM-Omni VRAM Orchestration Audit', fontsize=14, fontweight='bold')
    plt.xlabel('Time (Seconds)', fontsize=12)
    plt.ylabel('VRAM Usage (MiB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', shadow=True)
    
    output_png = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_png, dpi=300)
    print(f"物理绘图完成！图片已保存至: {output_png}")

if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) > 1 else 'vram_leak_check.csv'
    generate_vram_plot(file_name)