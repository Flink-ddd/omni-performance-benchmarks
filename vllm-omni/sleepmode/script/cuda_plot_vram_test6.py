import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_amd_perfect_plot():
    csv_file = "data/A6000/nvidia-6.csv"
    output_img = "vram_unitest_6.png"
    
    df = pd.read_csv(csv_file)
    df_gpu1 = df[df['device'] == 'card1'].copy().reset_index(drop=True)
    df_gpu1['vram_gib'] = df_gpu1['VRAM Total Used Memory (B)'] / (1024**3)

    peak_idx = df_gpu1['vram_gib'].idxmax()
    peak_val = df_gpu1.loc[peak_idx, 'vram_gib']

    after_peak = df_gpu1.loc[peak_idx:]
    sleep_candidates = after_peak[after_peak['vram_gib'] < 5.0]
    
    if sleep_candidates.empty:
        print("未发现睡眠点")
        return
        
    sleep_idx = sleep_candidates.index[0]
    sleep_val = df_gpu1.loc[sleep_idx, 'vram_gib']

    after_sleep = df_gpu1.loc[sleep_idx:]
    wake_candidates = after_sleep[after_sleep['vram_gib'] > 15.0]

    plt.figure(figsize=(13, 7), dpi=150)
    plt.plot(df_gpu1.index, df_gpu1['vram_gib'], color='#e63946', linewidth=2.5, label='AMD MI300X (Diffusion)')
    plt.fill_between(df_gpu1.index, df_gpu1['vram_gib'], color='#e63946', alpha=0.1)

    # Active
    plt.annotate(f'Active State\n({peak_val:.1f} GiB)', 
                 xy=(peak_idx, peak_val), xytext=(peak_idx - 10, peak_val - 5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1), ha='center', fontweight='bold')

    # Sleep (Reclaimed)
    plt.annotate(f'Sleep Mode\n(Reclaimed)', 
                 xy=(sleep_idx, sleep_val), xytext=(sleep_idx + 10, sleep_val - 5),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=1), 
                 color='#27ae60', fontweight='bold', ha='center')

    # Wakeup (Recovery)
    if not wake_candidates.empty:
        wake_idx = wake_candidates.index[0]
        wake_val = df_gpu1.loc[wake_idx, 'vram_gib']
        plt.annotate(f'Wakeup Success\n({wake_val:.1f} GiB)', 
                     xy=(wake_idx, wake_val), xytext=(wake_idx + 10, wake_val - 5),
                     arrowprops=dict(facecolor='#2980b9', shrink=0.05, width=1), 
                     color='#2980b9', fontweight='bold', ha='center')

    plt.title('AMD MI300X VRAM Lifecycle: GPU1 sleep mode level 2', fontsize=15, fontweight='bold')
    plt.ylabel('Physical VRAM Usage (GiB)')
    plt.xlabel('Time (Seconds)')
    plt.ylim(0, 45)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.savefig(output_img)
    print(f"图片已生成: {output_img}")

if __name__ == "__main__":
    generate_amd_perfect_plot()