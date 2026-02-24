import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def generate_universal_audit_plot(csv_file):
    if not os.path.exists(csv_file):
        print(f"错误：找不到文件 {csv_file}。请检查路径或直接传文件名。")
        return

    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    
    df['vram'] = df['memory.used [MiB]'].astype(str).str.replace(' MiB', '').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    gpu_stats = df.groupby('index')['vram'].std()
    target_gpu_index = gpu_stats.idxmax()
    gpu_name = df[df['index'] == target_gpu_index]['name'].iloc[0].strip()
    
    print(f"物理探测成功：检测到活跃显卡为 GPU {target_gpu_index} ({gpu_name})")
    
    data = df[df['index'] == target_gpu_index].copy().reset_index(drop=True)
    
    plt.figure(figsize=(13, 7), dpi=150)
    plt.plot(data['time'], data['vram'], color='#c0392b', linewidth=2.5, label=f'GPU {target_gpu_index} VRAM ({gpu_name})')

    peak_val = data['vram'].max()
    peak_idx = data['vram'].idxmax()
    peak_time = data.loc[peak_idx, 'time']
    
    after_peak = data[data['time'] > peak_time]
    sleep_candidates = after_peak[after_peak['vram'] < (peak_val * 0.2)]
    
    if not sleep_candidates.empty:
        sleep_idx = sleep_candidates.index[0]
        sleep_time = data.loc[sleep_idx, 'time']
        sleep_val = data.loc[sleep_idx, 'vram']
        
        after_sleep = data[data['time'] > sleep_time]
        wake_candidates = after_sleep[after_sleep['vram'] > (peak_val * 0.8)]
        
        if not wake_candidates.empty:
            wake_idx = wake_candidates.index[0]
            wake_time = data.loc[wake_idx, 'time']
            wake_val = data.loc[wake_idx, 'vram']
            
            plt.annotate(f'Wakeup Success\n({wake_val/1024:.1f} GiB)', 
                         xy=(wake_time, wake_val), xytext=(wake_time + 5, wake_val - 5000),
                         arrowprops=dict(facecolor='#2980b9', shrink=0.05, width=1), color='#2980b9', fontweight='bold')

        # 标注 Sleep
        plt.annotate(f'Sleep Mode\n(VRAM Reclaimed)', 
                     xy=(sleep_time, sleep_val), xytext=(sleep_time - 15, sleep_val + 3000),
                     arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=1), color='#27ae60', fontweight='bold')

    # 标注 Active
    plt.annotate(f'Active State\n({peak_val/1024:.1f} GiB)', 
                 xy=(peak_time, peak_val), xytext=(peak_time - 40, peak_val + 2000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.title(f'Universal VRAM Lifecycle Audit: {gpu_name} (A-Series Compatible)', fontsize=15, fontweight='bold')
    plt.ylabel('VRAM Usage (MiB)')
    plt.xlabel('Test Duration (Seconds)')
    plt.ylim(0, peak_val + 10000)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    output_name = f'vram_audit_{gpu_name.replace(" ", "_")}.png'
    plt.savefig(output_name, dpi=300)
    print(f"物理审计图已成功生成：{output_name}")

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else 'test6.csv'
    abs_path = os.path.abspath(fname)
    print(f"正在尝试读取物理路径: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"错误：在 {abs_path} 找不到文件！")
    else:
        generate_universal_audit_plot(abs_path)