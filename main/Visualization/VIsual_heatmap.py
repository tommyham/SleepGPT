import copy

import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import tensor
import pandas as pd
import seaborn as sns
from math import pi
from scipy.stats import ttest_ind, chi2_contingency

cmap = LinearSegmentedColormap.from_list("mycmap",
                                         [(0, "#282A62"), (0.2, "#692F7C"), (0.4, "#B43970"),
                                          (0.6, "#d96558"), (0.8, "#efa143"), (1, "#F6C63C")])
cmap2 = LinearSegmentedColormap.from_list("mycmap",
                                         [(0, "#9ec1d4"), (0.2, "#ddf1f3"), (0.4, "#ecf4dd"),
                                          (0.6, "#fff7ac"), (0.8, "#ecb477"), (1, "#e87651")])
def confusion(cm: torch.Tensor):
    sum0 = cm.sum(axis=0)
    sum1 = cm.sum(axis=1)
    all_sum = cm.sum()
    p0 = torch.diag(cm).sum() / all_sum
    FP = sum0 - torch.diag(cm)
    FN = sum1 - torch.diag(cm)
    TP = torch.diag(cm)
    acc = TP.sum()/all_sum

    TN = all_sum - FP - FN - TP

    precision = TP / (TP + FP + 1e-6)

    recall = TP / (TP + FN+ 1e-6)

    pe = (sum0 * sum1).sum() / (all_sum ** 2)

    kappa = (p0 - pe) / (1 - pe)
    sensitivity = TP/ (TP + FN)
    specificity = TN / (FP + TN)
    macro_f1_sub = 2 * precision * recall / (precision + recall+ 1e-6)
    macro_f1 = torch.mean(macro_f1_sub, dim=-1)
    return precision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub
def main():
    root_path = '/Users/hwx_admin/Sleep/DATA_result/result'
    dataset_path = glob.glob(os.path.join(root_path, '*'))
    ds_res = {}

    # F6C63C
    # efa143 # d96558 #B43970 #692F7C #282A62
    vmin, vmax = 100, 0
    for dataset in dataset_path:
        if 'EDF' in dataset:
            continue
        ds_res[os.path.basename(dataset)] = {}
        ckpt_path = glob.glob(os.path.join(dataset, 'attn/*'))
        res = {0: torch.zeros(8), 1: torch.zeros(8), 2: torch.zeros(8), 3: torch.zeros(8), 4: torch.zeros(8)}
        for ckpts in ckpt_path:
            ckpt = torch.load(os.path.join(ckpts, 'res.ckpt'), map_location='cpu')
            for k, v in ckpt.items():
                res[k] += v
        ds_res[os.path.basename(dataset)] = res
    heatmap_data_list = []
    all_heatmaps = {}
    for dataset_name in [1, 2, 3, 5]:
        data = ds_res[f'MASS{dataset_name}']

        heatmap_data = np.array([data[i].numpy() for i in data.keys()], dtype=np.float32)
        sum = np.array([heatmap_data[i].sum() for i in range(heatmap_data.shape[0])]).reshape(-1, 1)
        heatmap_data = heatmap_data/sum  # 20 fold
        heatmap_data_list.append(heatmap_data)
        vmin = min(vmin, heatmap_data.min())
        vmax = max(vmax, heatmap_data.max())
        all_heatmaps[f'MASS{dataset_name}'] = heatmap_data
    print(vmin, vmax)
    save_dir = '/Users/hwx_admin/Sleep/result/heatmap/doc/fig2'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'heatmap_data.xlsx')
    with pd.ExcelWriter(save_path) as writer:
        electrode_channels = ['C3', 'C4', 'EOG', 'EMG', 'F3', 'Fpz', 'O1', 'Pz']
        sleep_stages = ['W', 'Stage 1', 'Stage 2', 'Stage 3', 'REM']
        for dataset_name, heatmap_data in all_heatmaps.items():
            df = pd.DataFrame(
                heatmap_data,
                index=sleep_stages[:heatmap_data.shape[0]],
                columns=electrode_channels
            )
            df.to_excel(writer, sheet_name=dataset_name)
    # for i, dataset_name in enumerate([1, 2, 3, 5]):
    #     print(dataset_name, heatmap_data)
    #     heatmap_data = heatmap_data_list[i]
    #     # 使用Matplotlib创建热图
    #     plt.figure(figsize=(10, 5))
    #     ax = sns.heatmap(heatmap_data, cmap=cmap2, vmin=vmin, vmax=vmax)
    #     # plt.colorbar()  # 显示颜色条
    #     # plt.title('Heatmap of EEG Electrodes by Sleep Stage')
    #     plt.xlabel('Electrode Channels')
    #     plt.ylabel('Sleep Stage Label')
    #     electrode_channels = ['C3', 'C4', 'EOG', 'EMG', 'F3', 'Fpz', 'O1', 'Pz']
    #     sleep_stages = ['W', 'Stage 1', 'Stage 2', 'Stage 3', 'REM']
    #     ax.set_xticks(np.arange(len(electrode_channels)) + 0.5)
    #     ax.set_yticks(np.arange(len(sleep_stages)) + 0.5)
    #     ax.set_xticklabels(electrode_channels, rotation=0, ha="center")  # 水平居中对齐
    #     ax.set_yticklabels(sleep_stages, rotation=0, va="center")  # 垂直居中对齐
    #     plt.subplots_adjust(bottom=0.15, top=0.85)
    #     plt.savefig(f'/Users/hwx_admin/Sleep/result/heatmap/{dataset_name}_heatmap.svg')
    #     plt.show()

def visual_radar_f1():
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
    from torch import tensor

    # ====== 输入数据（5x5 混淆矩阵，行=真实，列=预测；两个模型 0/1；数据集 MASS1/2/3/5）======
    mass = [{0: None, 1: None} for _ in range(6)]
    mass[3][0] = tensor([[1964, 156,  64,   4, 137],
                         [ 105, 473, 484,   2, 505],
                         [  33, 103, 8043, 360, 238],
                         [   1,   0,  407, 2124,  6],
                         [  11,  93,  151,   0, 2816]])
    mass[3][1] = tensor([[2064, 114,  44,   3, 100],
                         [  77, 503, 500,   0, 489],
                         [  49,  98, 8047, 401, 182],
                         [   3,   0,  545, 1990,  0],
                         [  17,  72,  185,   2, 2795]])
    mass[5][0] = tensor([[ 978,   3, 104,   8,  91],
                         [ 125,  18, 233,   0, 319],
                         [  22,   3, 5343, 136,  95],
                         [   0,   0,  307, 1562,  0],
                         [  23,   0,  123,   0, 2307]])
    mass[5][1] = tensor([[1001,  49,  65,   7,  62],
                         [  65,  89, 210,   0, 331],
                         [  26,  26, 5337,  93, 117],
                         [   4,   0,  399, 1466,   0],
                         [  28,   4,  172,   1, 2248]])
    mass[1][0] = tensor([[2744,   38,  256,    0,   41],
                         [ 666,   49, 1217,    0,  309],
                         [ 136,   23, 6950,    0,  167],
                         [   0,    0, 1081,    0,    0],
                         [ 419,   20,  500,    0,  904]])
    mass[1][1] = tensor([[2917,  102,   14,    0,   46],
                         [ 413,  594,  784,    1,  449],
                         [  67,  337, 6455,  186,  231],
                         [   1,    0,  455,  625,    0],
                         [ 172,   77,  193,    0, 1401]])
    mass[2][1] = tensor([[388,   0,  29,   4, 464],
                         [  6,   0, 103,   1, 346],
                         [  9,   0, 3398, 436, 260],
                         [  1,   0, 103, 1212,   0],
                         [  7,   0,  99,   0, 1614]])
    mass[2][0] = tensor([[  0,   0,  191,   9, 685],
                         [  0,   0,   88,   1, 367],
                         [  0,   0, 3039, 637, 427],
                         [  0,   0,   90, 1226,   0],
                         [  0,   0,  125,   1, 1594]])

    # 画图风格
    plt.style.use('ggplot')

    # ====== 汇总 overall（按样本数相加）======
    res_overall = [mass[1][0].detach().clone(), mass[1][1].detach().clone()]
    for _i in [2, 3, 5]:
        res_overall[0] += mass[_i][0]
        res_overall[1] += mass[_i][1]

    # ====== 计算 overall 的每类 F1（原始，未归一化）======
    # 依赖你已有的 confusion(cm) 函数：返回 precision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub
    ans = [0 for _ in range(2)]
    overall_metric = [0, 0]
    for i in range(2):
        precision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub = confusion(res_overall[i])
        ans[i] = macro_f1_sub.detach().cpu().numpy()  # 原始每类 F1（未归一化）
        overall_metric[i] = (kappa.item(), acc.item(), macro_f1.item())

    # ====== 分数据集计算 kappa/acc/mf1，便于输出“4个值+统计”======
    overall_kappa, overall_acc, overall_macro_f1 = [[], []], [[], []], [[], []]
    for _i in [1, 2, 3, 5]:
        for model in range(2):
            precision, recall, kappa, sensitivity, specificity, acc, macro_f1, macro_f1_sub = confusion(mass[_i][model])
            overall_kappa[model].append(kappa.item())
            overall_acc[model].append(acc.item())
            overall_macro_f1[model].append(macro_f1.item())

    # 取四个值的均值作为 overall 的“平均指标”（仅用于展示）
    for model in range(2):
        overall_metric[model] = (
            np.mean(overall_kappa[model]),
            np.mean(overall_acc[model]),
            np.mean(overall_macro_f1[model])
        )

    # 卡方与其他统计
    chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(np.stack([res_overall[0].numpy(), res_overall[1].numpy()]).sum(axis=0))
    kappa_p_value = ttest_ind(overall_kappa[0], overall_kappa[1], equal_var=False).pvalue
    acc_p_value   = ttest_ind(overall_acc[0],   overall_acc[1],   equal_var=False).pvalue
    mf1_p_value   = ttest_ind(overall_macro_f1[0], overall_macro_f1[1], equal_var=False).pvalue
    mw_stat, mw_p = mannwhitneyu(overall_acc[0], overall_acc[1])

    # ====== 为雷达图制作“归一化”的每类F1（逐类对两模型取 max 归一化）======
    # 先分数据集收集原始与归一化 F1
    stage_labels = ['W', 'N1', 'N2', 'N3', 'REM']
    per_dataset_f1_raw  = []  # [{0: raw_m0, 1: raw_m1}, ...] 按 MASS1/2/3/5 顺序
    per_dataset_f1_norm = []
    res = []  # 雷达图用（归一化）

    for index in [1, 2, 3, 5]:
        _, _, _, _, _, _, _, macro_f1_sub_m0 = confusion(mass[index][0])
        raw_m0 = macro_f1_sub_m0.detach().cpu().numpy()
        _, _, _, _, _, _, _, macro_f1_sub_m1 = confusion(mass[index][1])
        raw_m1 = macro_f1_sub_m1.detach().cpu().numpy()

        per_dataset_f1_raw.append({0: raw_m0.copy(), 1: raw_m1.copy()})
        max_per_class = np.maximum(raw_m0, raw_m1)
        norm_m0 = np.where(max_per_class != 0, raw_m0 / max_per_class, 0.0)
        norm_m1 = np.where(max_per_class != 0, raw_m1 / max_per_class, 0.0)
        per_dataset_f1_norm.append({0: norm_m0, 1: norm_m1})
        res.append({0: norm_m0, 1: norm_m1})

    # overall 的原始&归一化 F1
    ans_raw_m0 = ans[0].copy()
    ans_raw_m1 = ans[1].copy()
    max_per_class_overall = np.maximum(ans_raw_m0, ans_raw_m1)
    ans_norm_m0 = np.where(max_per_class_overall != 0, ans_raw_m0 / max_per_class_overall, 1.0)
    ans_norm_m1 = np.where(max_per_class_overall != 0, ans_raw_m1 / max_per_class_overall, 1.0)
    res.append({0: ans_norm_m0, 1: ans_norm_m1})

    # ====== 画雷达图（使用归一化后的）======
    df = pd.DataFrame(res)
    df.index = ['1', '2', '3', '4', '5']  # 4 个数据集 + OVERALL
    num_vars = 5
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for i, row in df.iterrows():
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        values0 = np.append(row[0], row[0][0])
        values1 = np.append(row[1], row[1][0])

        ax.plot(angles, values0, '#4DBADB', linewidth=2, label='Model 0')
        ax.fill(angles, values0, '#4DBADB', alpha=0.25)
        ax.scatter(angles, values0, color='#4DBADB', marker='o')

        ax.plot(angles, values1, '#E44A33', linewidth=2, label='Model 1')
        ax.fill(angles, values1, '#E44A33', alpha=0.25)
        ax.scatter(angles, values1, color='#E44A33', marker='o')

        ax.set_thetagrids(np.degrees(angles[:-1]), stage_labels)
        ax.set_rlabel_position(90)
        ax.set_rticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylim([0, 1.1])
        ax.set_yticklabels([])
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
        plt.savefig(f'/Users/hwx_admin/Sleep/result/heatmap/{i}.svg')
        plt.close(fig)

    # ====== 总体条形图（均值+误差条+散点）======
    categories = ["Kappa", "Accuracy", "Macro-F1"]
    x = np.arange(len(categories))
    width = 0.35

    # 计算均值与标准差
    model0_mean = [np.mean(overall_kappa[0]), np.mean(overall_acc[0]), np.mean(overall_macro_f1[0])]
    model1_mean = [np.mean(overall_kappa[1]), np.mean(overall_acc[1]), np.mean(overall_macro_f1[1])]
    model0_std = [np.std(overall_kappa[0]), np.std(overall_acc[0]), np.std(overall_macro_f1[0])]
    model1_std = [np.std(overall_kappa[1]), np.std(overall_acc[1]), np.std(overall_macro_f1[1])]

    fig, ax = plt.subplots(figsize=(7, 4))

    # 绘制 bar（带误差条）
    rects1 = ax.barh(x - width / 2, model0_mean, width, xerr=model0_std,
                     label='Model 0', color='#4DBADB', alpha=0.8, capsize=4, edgecolor='black', linewidth=0.6)
    rects2 = ax.barh(x + width / 2, model1_mean, width, xerr=model1_std,
                     label='Model 1', color='#E44A33', alpha=0.8, capsize=4, edgecolor='black', linewidth=0.6)

    # 叠加每个数据集的原始点（散点）
    jitter = 0.07
    for i in range(len(categories)):
        # 各数据集的4个值散点（model 0）
        x0_vals = [overall_kappa[0], overall_acc[0], overall_macro_f1[0]][i]
        y0_pos = np.random.normal(x[i] - width / 2, jitter, len(x0_vals))
        ax.scatter(x0_vals, y0_pos, color='black', s=18, alpha=0.6, zorder=3)

        # 各数据集的4个值散点（model 1）
        x1_vals = [overall_kappa[1], overall_acc[1], overall_macro_f1[1]][i]
        y1_pos = np.random.normal(x[i] + width / 2, jitter, len(x1_vals))
        ax.scatter(x1_vals, y1_pos, color='black', s=18, alpha=0.6, zorder=3)

    # 坐标与样式
    ax.set_yticks(x)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlim([0.4, 0.9])
    ax.set_xlabel("Score", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    plt.tight_layout()

    plt.savefig('/Users/hwx_admin/Sleep/result/heatmap/overall_metric_points.svg', dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ====== 导出 Excel：包含原始/归一化 F1、四个值及 mean/std、统计检验 ======
    try:
        import openpyxl  # noqa
    except ImportError:
        raise ImportError("需要安装 openpyxl：`pip install openpyxl`")

    save_dir = "/Users/hwx_admin/Sleep/result/heatmap/doc/fig2"
    os.makedirs(save_dir, exist_ok=True)
    xlsx_path = os.path.join(save_dir, "visual_radar_f1_out.xlsx")

    # overall 混淆矩阵
    stage_labels = ['W', 'N1', 'N2', 'N3', 'REM']
    overall0_df = pd.DataFrame(res_overall[0].detach().cpu().numpy(), index=stage_labels, columns=stage_labels)
    overall1_df = pd.DataFrame(res_overall[1].detach().cpu().numpy(), index=stage_labels, columns=stage_labels)

    # overall 指标（均值）
    overall_met_df = pd.DataFrame({
        "model": ["model0", "model1"],
        "kappa_mean": [overall_metric[0][0], overall_metric[1][0]],
        "acc_mean":   [overall_metric[0][1], overall_metric[1][1]],
        "mf1_mean":   [overall_metric[0][2], overall_metric[1][2]],
    })

    # 每个数据集上的 4 个值 + mean/std（t 检验用的“4 个值”也从这里取）
    per_ds_values_df = []
    for m in (0, 1):
        vals = pd.DataFrame({
            "dataset": ["MASS1", "MASS2", "MASS3", "MASS5"],
            "model":   f"model{m}",
            "kappa":   overall_kappa[m],
            "acc":     overall_acc[m],
            "mf1":     overall_macro_f1[m],
        })
        vals["kappa_mean"] = np.mean(overall_kappa[m])
        vals["kappa_std"]  = np.std(overall_kappa[m], ddof=1)
        vals["acc_mean"]   = np.mean(overall_acc[m])
        vals["acc_std"]    = np.std(overall_acc[m], ddof=1)
        vals["mf1_mean"]   = np.mean(overall_macro_f1[m])
        vals["mf1_std"]    = np.std(overall_macro_f1[m], ddof=1)
        per_ds_values_df.append(vals)
    per_ds_values_df = pd.concat(per_ds_values_df, ignore_index=True)

    # “原始”每类F1（未归一化）
    raw_rows = []
    for di, d in enumerate([1, 2, 3, 5], start=1):
        f1_m0 = per_dataset_f1_raw[di-1][0]
        f1_m1 = per_dataset_f1_raw[di-1][1]
        raw_rows.append(pd.DataFrame({
            "dataset": f"MASS{d}",
            "class": stage_labels,
            "model0_f1": f1_m0,
            "model1_f1": f1_m1,
        }))
    raw_rows.append(pd.DataFrame({
        "dataset": "OVERALL",
        "class": stage_labels,
        "model0_f1": ans_raw_m0,
        "model1_f1": ans_raw_m1,
    }))
    raw_f1_df = pd.concat(raw_rows, ignore_index=True)

    # “归一化”每类F1（逐类在两模型间归一化）
    norm_rows = []
    for di, d in enumerate([1, 2, 3, 5], start=1):
        f1_m0 = per_dataset_f1_norm[di-1][0]
        f1_m1 = per_dataset_f1_norm[di-1][1]
        norm_rows.append(pd.DataFrame({
            "dataset": f"MASS{d}",
            "class": stage_labels,
            "model0_f1_norm": f1_m0,
            "model1_f1_norm": f1_m1,
        }))
    norm_rows.append(pd.DataFrame({
        "dataset": "OVERALL",
        "class": stage_labels,
        "model0_f1_norm": ans_norm_m0,
        "model1_f1_norm": ans_norm_m1,
    }))
    norm_f1_df = pd.concat(norm_rows, ignore_index=True)

    # 统计检验 + t 检验 4 个值明细
    stats_df = pd.DataFrame({
        "test": ["t-test (kappa)", "t-test (acc)", "t-test (mf1)",
                 "Mann-Whitney U (acc)", "Chi-square (overall confusion)"],
        "statistic": [None, None, None, mw_stat, chi2_stat],
        "p_value":   [kappa_p_value, acc_p_value, mf1_p_value, mw_p, chi2_p],
        "notes": ["equal_var=False", "equal_var=False", "equal_var=False",
                  "two-sided", "scipy.stats.chi2_contingency on pooled confusion"],
    })

    tt_values_df = pd.DataFrame({
        "metric": (["kappa"]*8 + ["acc"]*8 + ["mf1"]*8),
        "model":  (["model0"]*4 + ["model1"]*4)*3,
        "dataset": (["MASS1","MASS2","MASS3","MASS5"]*2)*3,
        "value":  overall_kappa[0] + overall_kappa[1] + overall_acc[0] + overall_acc[1] + overall_macro_f1[0] + overall_macro_f1[1],
    })

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # overall 混淆矩阵（左右并列）
        overall0_df.to_excel(writer, sheet_name="overall_confusions", startrow=1, startcol=0)
        overall1_df.to_excel(writer, sheet_name="overall_confusions", startrow=1, startcol=8)
        pd.DataFrame([["model0"]]).to_excel(writer, sheet_name="overall_confusions", startrow=0, startcol=0, header=False, index=False)
        pd.DataFrame([["model1"]]).to_excel(writer, sheet_name="overall_confusions", startrow=0, startcol=8, header=False, index=False)

        overall_met_df.to_excel(writer, sheet_name="overall_metrics_mean", index=False)
        per_ds_values_df.to_excel(writer, sheet_name="overall_metrics_values", index=False)

        raw_f1_df.to_excel(writer, sheet_name="perclassF1_raw", index=False)     # 未归一化
        norm_f1_df.to_excel(writer, sheet_name="perclassF1_norm", index=False)   # 归一化

        stats_df.to_excel(writer, sheet_name="stats", index=False)
        tt_values_df.to_excel(writer, sheet_name="t_test_values", index=False)

    print(f"✅ 原始/归一化 F1，四个值及 mean/std，统计检验均已保存：{xlsx_path}")

#plt
if __name__ == '__main__':
    main()
    # visual_radar_f1()