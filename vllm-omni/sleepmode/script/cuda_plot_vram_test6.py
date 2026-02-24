import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_perfect_audit_plot(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    df['vram'] = df['memory.used [MiB]'].str.replace(' MiB', '').astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    gpu1 = df[df['index'] == 1].copy().reset_index(drop=True)
    
    plt.figure(figsize=(13, 7))
    plt.plot(gpu1['time'], gpu1['vram'], color='#c0392b', linewidth=2.5, label='GPU 1 VRAM (Diffusion)')

    peak_idx = gpu1['vram'].idxmax()
    peak_val = gpu1.loc[peak_idx, 'vram']
    peak_time = gpu1.loc[peak_idx, 'time']
    
    after_peak = gpu1[gpu1['time'] > peak_time]
    sleep_candidates = after_peak[after_peak['vram'] < 2000]
    
    if sleep_candidates.empty:
        print("未发现明显的睡眠掉落点")
        return

    sleep_idx = sleep_candidates.index[0]
    sleep_val = gpu1.loc[sleep_idx, 'vram']
    sleep_time = gpu1.loc[sleep_idx, 'time']
    
    after_sleep = gpu1[gpu1['time'] > sleep_time]
    wake_candidates = after_sleep[after_sleep['vram'] > 10000]
    
    if not wake_candidates.empty:
        wake_idx = wake_candidates.index[0]
        wake_val = gpu1.loc[wake_idx, 'vram']
        wake_time = gpu1.loc[wake_idx, 'time']
        
        plt.annotate(f'Wakeup Success\n({wake_val/1024:.1f} GiB)', 
                     xy=(wake_time, wake_val), xytext=(wake_time + 8, wake_val - 5000),
                     arrowprops=dict(facecolor='#2980b9', shrink=0.05, width=1), color='#2980b9', fontweight='bold')
    else:
        print("未发现唤醒回升点")

    # 标注 Active
    plt.annotate(f'Active State\n({peak_val/1024:.1f} GiB)', 
                 xy=(peak_time, peak_val), xytext=(peak_time - 40, peak_val + 2000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # 标注 Sleep
    plt.annotate(f'Sleep Mode\n(Reclaimed)', 
                 xy=(sleep_time, sleep_val), xytext=(sleep_time - 20, sleep_val + 5000),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=1), color='#27ae60', fontweight='bold')

    plt.title('VRAM Lifecycle Audit: Diffusion Sleep & Wakeup Performance', fontsize=15, fontweight='bold')
    plt.ylabel('VRAM Usage (MiB)')
    plt.xlabel('Test Duration (Seconds)')
    plt.ylim(0, 26000)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    output_name = 'vram_perfect_lifecycle_v2.png'
    plt.savefig(output_name, dpi=300)
    print(f"物理审计图已成功生成：{output_name}")

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else 'test6.csv'
    generate_perfect_audit_plot(fname)