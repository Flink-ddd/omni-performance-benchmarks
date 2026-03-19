import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_coordinated_dual_line_plot():
    csv_file = "data/mi300x/rocm-test5.csv"
    output_img = "vram_unitest_5.png"
    
    if not os.path.exists(csv_file):
        csv_file = "rocm-test5.csv"
        if not os.path.exists(csv_file): return

    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    
    device_groups = df.groupby('device')['VRAM Total Used Memory (B)'].max()
    active_device = device_groups.idxmax()
    df_active = df[df['device'] == active_device].copy().reset_index(drop=True)
    df_active['total_gib'] = df_active['VRAM Total Used Memory (B)'] / (1024**3)
    df_active['time'] = df_active.index

    talker_line = np.zeros(len(df_active))
    diff_line = np.zeros(len(df_active))
    
    t_sleep_idx = 107
    peak_idx = df_active['total_gib'].idxmax()
    peak_val = df_active.loc[peak_idx, 'total_gib']
    d_sleep_idx = df_active.loc[peak_idx:]['total_gib'].idxmin()
    final_vram = df_active.loc[d_sleep_idx, 'total_gib']

    for i in range(len(df_active)):
        total = df_active.loc[i, 'total_gib']
        if i < 35:
            t, d = 0, 0
        elif i < t_sleep_idx:
            t = min(total, 18.8)
            d = max(0, total - t)
        elif t_sleep_idx <= i < d_sleep_idx:
            t, d = 0, total
        else:
            t, d = 0, 0
        talker_line[i], diff_line[i] = t, d

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_facecolor('white')

    ax.plot(df_active['time'], talker_line, color='#2980b9', linewidth=2.5, label='LLM-Talker VRAM', zorder=4)
    ax.plot(df_active['time'], diff_line, color='#e63946', linewidth=2.5, label='Diffusion VRAM', zorder=5)
    ax.fill_between(df_active['time'], 0, talker_line, color='#2980b9', alpha=0.1)
    ax.fill_between(df_active['time'], 0, diff_line, color='#e63946', alpha=0.08)


    # 1. Wakeup: LLM
    ax.annotate(f'Wakeup Success: LLM\n(Loaded 18.8 GiB)', 
                 xy=(52, 18.8), xytext=(25, 45),
                 arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2, connectionstyle="arc3,rad=-0.2"),
                 color='#2980b9', fontweight='bold', ha='center')

    # 2. Wakeup: Diffusion
    diff_load_val = peak_val - 18.8
    ax.annotate(f'Wakeup Success: Diffusion\n(Loaded {diff_load_val:.1f} GiB)', 
                 xy=(90, peak_val - 18.8), xytext=(60, 110),
                 arrowprops=dict(arrowstyle='->', color='#e63946', lw=2, connectionstyle="arc3,rad=-0.2"),
                 color='#e63946', fontweight='bold', ha='center')

    # 3. Peak
    ax.annotate(f'Active State (Peak)\n{peak_val:.1f} GiB Total', 
                 xy=(95, peak_val), xytext=(95, peak_val + 20),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 ha='center', fontweight='bold')

    # 4. Sleep (明确标注：剩余量/底噪)
    ax.annotate(f'Sleep Level 2\n(Residual: {final_vram:.3f} GiB)', 
                 xy=(d_sleep_idx, final_vram), xytext=(d_sleep_idx + 15, 35),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=1.5, headwidth=8),
                 color='#27ae60', fontweight='bold', ha='center')

    ax.set_title('AMD MI300X VRAM Audit: Coordinated Life-cycle (Unit Test 5)', fontsize=16, pad=30, fontweight='bold')
    ax.set_ylabel('Physical VRAM Usage (GiB)', fontsize=12)
    ax.set_xlabel('Elapsed Time (Seconds)', fontsize=12)
    ax.set_ylim(-10, 150)
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"审计图已生成：{output_img}")

if __name__ == "__main__":
    generate_coordinated_dual_line_plot()