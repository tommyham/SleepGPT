import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


def calculate_metrics(conf_matrix):
    y_true = []
    y_pred = []
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            y_true.extend([i] * conf_matrix[i, j])
            y_pred.extend([j] * conf_matrix[i, j])

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    print(f1)
    return acc, mf1, kappa, f1

def main():
    confusion_matrix_raw = {
        1: np.array([[0, 0, 1675, 0, 0], [0, 0, 621, 0, 0], [0, 0, 4206, 0, 0], [0, 0, 850, 0, 0], [0, 0, 1848, 0, 0]]),
        2: np.array([[0, 0, 1675, 0, 0], [0, 0, 621, 0, 0], [0, 0, 4206, 0, 0], [0, 0, 850, 0, 0], [0, 0, 1848, 0, 0]]),
        5: np.array([[1559, 4, 43, 7, 62], [200, 18, 102, 5, 296], [23, 3, 3682, 237, 261], [2, 0, 103, 745, 0],
                     [67, 8, 240, 0, 1533]]),
        12: np.array([[1583, 50, 14, 2, 26], [176, 204, 93, 3, 145], [48, 86, 3696, 164, 212], [9, 0, 132, 709, 0],
                      [51, 80, 146, 0, 1571]])
    }

    confusion_matrix_aug = {
        1: np.array([[424, 0, 1118, 133, 0], [6, 0, 609, 6, 0], [187, 0, 3829, 190, 0], [692, 0, 69, 89, 0],
                     [0, 0, 1845, 3, 0]]),
        2: np.array([[1550, 0, 125, 0, 0], [223, 0, 398, 0, 0], [466, 0, 3740, 0, 0], [633, 0, 217, 0, 0],
                     [359, 0, 1489, 0, 0]]),
        5: np.array([[1525, 32, 35, 3, 80], [151, 74, 118, 3, 275], [31, 11, 3690, 179, 295], [2, 0, 185, 663, 0],
                     [14, 17, 154, 0, 1663]]),
        12: np.array([[1559, 57, 37, 3, 19], [151, 210, 123, 2, 135], [20, 46, 3839, 177, 124], [2, 0, 135, 713, 0],
                      [32, 83, 163, 0, 1570]])
    }

    metrics_raw = {n: calculate_metrics(cm) for n, cm in confusion_matrix_raw.items()}
    metrics_aug = {n: calculate_metrics(cm) for n, cm in confusion_matrix_aug.items()}

    subject_counts = sorted(metrics_raw.keys())
    metric_names = ['ACC', 'MF1', 'Kappa'] + [f'Stage{i + 1}_F1' for i in range(5)]

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20), sharey=True)
    axes = axes.flatten()

    width = 0.35  # width of the bars

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        if i < 3:
            raw_values = [metrics_raw[n][i] for n in subject_counts]
            augmented_values = [metrics_aug[n][i] for n in subject_counts]
            for n in subject_counts:
                print(f'n: {n} augmented_values-raw_values: {metrics_aug[n][i]-metrics_raw[n][i]}')
        else:
            stage_idx = i - 3
            raw_values = [metrics_raw[n][3][stage_idx] for n in subject_counts]
            augmented_values = [metrics_aug[n][3][stage_idx] for n in subject_counts]

        x = np.arange(len(subject_counts))

        ax.bar(x - width/2, raw_values, width, label='Raw Data' if i == 0 else "")
        ax.bar(x + width/2, augmented_values, width, label='Augmented Data' if i == 0 else "")

        ax.set_title(f'{metric}')
        ax.set_xticks(x)
        ax.set_xticklabels(subject_counts)
        if i % 2 == 0:
            ax.set_ylabel('Score')
        if i >= len(metric_names) - 2:
            ax.set_xlabel('Number of Subjects')

    axes[0].legend(loc='upper right', bbox_to_anchor=(1.5, 1))

    fig.suptitle('Comparison of Raw Data and Augmented Data across Different Subject Counts')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../result/portion.svg')

    plt.show()
    save_all_metrics(metrics_raw, metrics_aug)

import os
import numpy as np
import pandas as pd

def save_all_metrics(metrics_raw, metrics_aug,
                     out_dir="/Users/hwx_admin/Sleep/result/heatmap/doc/fig2",
                     filename="portion_metrics.xlsx",
                     decimals=6):
    """
    metrics_raw / metrics_aug:
      dict: {subjects_count: (acc, mf1, kappa, f1_array_len5)}

    导出：
      - wide（宽表，含 raw/aug/delta）
      - long（长表，含 variant=raw/aug/delta）
      - barplot_source（与你当前画图用的数据一致）
    """
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, filename)

    subject_counts = sorted(metrics_raw.keys())
    metric_names = ['ACC', 'MF1', 'Kappa'] + [f'Stage{i+1}_F1' for i in range(5)]

    # ---------- 组装 wide（每个 subjects 一行） ----------
    rows = []
    for n in subject_counts:
        acc_r, mf1_r, kap_r, f1_r = metrics_raw[n]
        acc_a, mf1_a, kap_a, f1_a = metrics_aug[n]

        row = {
            "subjects": n,
            "ACC_raw": acc_r, "ACC_aug": acc_a, "ACC_delta": acc_a - acc_r,
            "MF1_raw": mf1_r, "MF1_aug": mf1_a, "MF1_delta": mf1_a - mf1_r,
            "Kappa_raw": kap_r, "Kappa_aug": kap_a, "Kappa_delta": kap_a - kap_r,
        }
        # 5 个阶段的 F1
        for i in range(5):
            row[f"Stage{i+1}_F1_raw"] = f1_r[i]
            row[f"Stage{i+1}_F1_aug"] = f1_a[i]
            row[f"Stage{i+1}_F1_delta"] = f1_a[i] - f1_r[i]

        rows.append(row)

    wide_df = pd.DataFrame(rows).sort_values("subjects").reset_index(drop=True)
    if decimals is not None:
        num_cols = [c for c in wide_df.columns if c != "subjects"]
        wide_df[num_cols] = wide_df[num_cols].astype(float).round(decimals)

    # ---------- 组装 long（含 raw/aug/delta 三种 variant） ----------
    long_rows = []
    for n in subject_counts:
        acc_r, mf1_r, kap_r, f1_r = metrics_raw[n]
        acc_a, mf1_a, kap_a, f1_a = metrics_aug[n]

        values = {
            "ACC": (acc_r, acc_a),
            "MF1": (mf1_r, mf1_a),
            "Kappa": (kap_r, kap_a),
        }
        for i in range(5):
            values[f"Stage{i+1}_F1"] = (f1_r[i], f1_a[i])

        for metric, (vr, va) in values.items():
            long_rows.append({"subjects": n, "metric": metric, "variant": "raw", "value": vr})
            long_rows.append({"subjects": n, "metric": metric, "variant": "aug", "value": va})
            long_rows.append({"subjects": n, "metric": metric, "variant": "delta", "value": va - vr})

    long_df = pd.DataFrame(long_rows).sort_values(["metric", "subjects", "variant"]).reset_index(drop=True)
    if decimals is not None:
        long_df["value"] = long_df["value"].astype(float).round(decimals)

    # ---------- barplot_source（与你的绘图一致） ----------
    # 前三项是 ACC/MF1/Kappa；后面 5 项是 Stage1..5 F1
    bar_rows = []
    for i, metric in enumerate(metric_names):
        for n in subject_counts:
            if i < 3:
                v_raw = metrics_raw[n][i]
                v_aug = metrics_aug[n][i]
            else:
                idx = i - 3
                v_raw = metrics_raw[n][3][idx]
                v_aug = metrics_aug[n][3][idx]
            bar_rows.append({"metric": metric, "subjects": n, "variant": "raw", "value": v_raw})
            bar_rows.append({"metric": metric, "subjects": n, "variant": "aug", "value": v_aug})
            bar_rows.append({"metric": metric, "subjects": n, "variant": "delta", "value": v_aug - v_raw})
    barplot_source = pd.DataFrame(bar_rows).sort_values(["metric", "subjects", "variant"]).reset_index(drop=True)
    if decimals is not None:
        barplot_source["value"] = barplot_source["value"].astype(float).round(decimals)

    # ---------- 写盘 ----------
    try:
        import openpyxl  # noqa
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as wr:
            wide_df.to_excel(wr, index=False, sheet_name="wide")
            long_df.to_excel(wr, index=False, sheet_name="long")
            barplot_source.to_excel(wr, index=False, sheet_name="barplot_source")
        print(f"[Saved] {xlsx_path}")
    except ImportError:
        wide_df.to_csv(os.path.join(out_dir, "portion_metrics_wide.csv"), index=False)
        long_df.to_csv(os.path.join(out_dir, "portion_metrics_long.csv"), index=False)
        barplot_source.to_csv(os.path.join(out_dir, "portion_metrics_barplot_source.csv"), index=False)
        print(f"[Saved CSVs] in {out_dir}")

if __name__ == "__main__":
    main()
