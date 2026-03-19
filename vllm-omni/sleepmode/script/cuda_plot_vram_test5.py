import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')

def generate_coordinated_dual_line_plot_ultimate():
    csv_file = "data/A6000/nvidia-5.csv"
    output_img = "vram_unitest_5_nvdia_aligned_red.png"
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到输入文件: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    
    df_active = df[df['index'] == 1].copy().reset_index(drop=True)
    df_active['total_gib'] = df_active['memory.used [MiB]'].str.replace(' MiB', '').astype(float) / 1024
    df_active['time'] = range(len(df_active))

    talker_line = np.zeros(len(df_active))
    diff_line = np.zeros(len(df_active))
    
    peak_val = df_active['total_gib'].max()
    t_sleep_idx = 93
    d_sleep_idx = 94

    talker_weight = 18.8 

    for i in range(len(df_active)):
        total = df_active.loc[i, 'total_gib']
        if i < 40:
            t, d = 0, 0
        elif i < t_sleep_idx:
            t = min(total, talker_weight)
            d = max(0, total - t)
        elif t_sleep_idx <= i < d_sleep_idx:
            # 协同睡眠：Talker 已卸载
            t, d = 0, total
        else:
            t, d = 0, total
            
        talker_line[i], diff_line[i] = t, d

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_facecolor('white')

    # 画线
    ax.plot(df_active['time'], talker_line, color='#2980b9', linewidth=3, label='LLM-Talker VRAM', zorder=4)
    ax.plot(df_active['time'], diff_line, color='#e63946', linewidth=3, label='Diffusion VRAM', zorder=5)
    
    # 填充
    ax.fill_between(df_active['time'], 0, talker_line, color='#2980b9', alpha=0.15)
    ax.fill_between(df_active['time'], 0, diff_line, color='#e63946', alpha=0.1)


    # 指向 LLM 平台 (18.8G)
    ax.annotate(f'Wakeup Success: LLM\n({talker_weight:.1f} GiB)', 
                 xy=(48, talker_weight), xytext=(30, 35),
                 arrowprops=dict(arrowstyle='->', color='#2980b9', lw=2, connectionstyle="arc3,rad=0.2"),
                 color='#2980b9', fontweight='bold', ha='center')

    diff_comp_height = peak_val - talker_weight
    ax.annotate(f'Wakeup Success: Diffusion\n({diff_comp_height:.1f} GiB)', 
                 xy=(90, diff_comp_height),
                 xytext=(100, 35),
                 arrowprops=dict(arrowstyle='->', color='#e63946', lw=2),
                 color='#e63946', fontweight='bold', ha='center')

    ax.annotate(f'Active State (Peak)\n{peak_val:.1f} GiB Total', 
                 xy=(90, peak_val), xytext=(110, 48),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 ha='center', fontweight='bold')

    # 协同睡眠
    final_vram = df_active.loc[d_sleep_idx, 'total_gib']
    ax.annotate(f'Coordinated Sleep\n(Residual: {final_vram:.2f} GiB)', 
                 xy=(d_sleep_idx, final_vram), xytext=(d_sleep_idx + 15, 10),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=1.5, headwidth=8),
                 color='#27ae60', fontweight='bold', ha='center')

    ax.set_title('NVIDIA RTX A6000 VRAM Audit: Coordinated Life-cycle (Unit Test 5)', fontsize=16, pad=35, fontweight='bold')
    ax.set_ylabel('Physical VRAM Usage (GiB)', fontsize=12)
    ax.set_xlabel('Elapsed Time (Seconds)', fontsize=12)
    ax.set_ylim(-2, 65)
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"专业校准版审计图已生成：{output_img}")

if __name__ == "__main__":
    generate_coordinated_dual_line_plot_ultimate()