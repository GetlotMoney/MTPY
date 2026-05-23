"""
α-grid search: 跑 6 个 α 值的加权融合实验
=========================================
对每个 α 值修改 yaml 然后跑 train_VGSR_CUB.py，
最后整理 best results 为对照表。

用法:
    F:\\Anaconda\\envs\\dassl_clip\\python.exe tools/sweep_text_alpha.py
"""
import os, re, subprocess, time, sys
import yaml
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, 'config', 'VGSR_cub_gzsl.yaml')
TRAIN_SCRIPT = os.path.join(ROOT, 'train_VGSR_CUB.py')
PYTHON = r'F:\Anaconda\envs\dassl_clip\python.exe'
LOG_DIR = os.path.join(ROOT, 'train_log', 'CUB')

# 要跑的 α 值
ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

# 备份原 yaml
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    original_yaml = f.read()


def update_yaml(alpha):
    """修改 yaml 中的 text_source 和 text_alpha"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['text_source'] = {'value': 'weighted'}
    cfg['text_alpha'] = {'value': alpha}
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def get_latest_log():
    """找最新生成的训练日志"""
    files = [f for f in os.listdir(LOG_DIR) if f.startswith('training_log_CUB_') and f.endswith('.txt')]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)), reverse=True)
    return os.path.join(LOG_DIR, files[0])


def parse_best(log_path):
    """从日志末尾解析 Best Results"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.search(
        r'Best Results @ Epoch\s+(\d+).*?'
        r'GZSL-U\s*:\s*([\d.]+)%.*?'
        r'GZSL-S\s*:\s*([\d.]+)%.*?'
        r'GZSL-H\s*:\s*([\d.]+)%.*?'
        r'ZSL\s*:\s*([\d.]+)%',
        content, re.DOTALL
    )
    if m:
        return {
            'epoch': int(m.group(1)),
            'U': float(m.group(2)),
            'S': float(m.group(3)),
            'H': float(m.group(4)),
            'ZSL': float(m.group(5)),
        }
    return None


def run_one(alpha):
    print(f"\n{'='*70}")
    print(f"  Running α = {alpha}")
    print(f"{'='*70}")
    update_yaml(alpha)

    start_time = time.time()
    result = subprocess.run([PYTHON, TRAIN_SCRIPT],
                            cwd=ROOT, capture_output=False)
    elapsed = time.time() - start_time
    print(f"  Elapsed: {elapsed/60:.1f} min")

    log_path = get_latest_log()
    best = parse_best(log_path)
    print(f"  Log: {os.path.basename(log_path)}")
    print(f"  Best: {best}")
    return best, log_path


def main():
    results = {}
    overall_start = time.time()

    for alpha in ALPHAS:
        best, log_path = run_one(alpha)
        results[alpha] = {'best': best, 'log': os.path.basename(log_path)}

        # 中间保存到 csv（防意外）
        save_results(results)

    # 恢复原 yaml
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(original_yaml)

    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"  Sweep complete in {total/60:.1f} min")
    print(f"{'='*70}")

    print_table(results)


def save_results(results):
    """保存到 CSV 文件"""
    csv_path = os.path.join(ROOT, 'train_log', 'CUB', 'sweep_text_alpha_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('alpha,U,S,H,ZSL,best_epoch,log\n')
        for alpha, r in sorted(results.items()):
            b = r['best']
            if b:
                f.write(f"{alpha},{b['U']:.2f},{b['S']:.2f},{b['H']:.2f},{b['ZSL']:.2f},{b['epoch']},{r['log']}\n")


def print_table(results):
    print("\n  α-grid search 结果:")
    print(f"  {'α':>6} | {'U':>6} | {'S':>6} | {'H':>6} | {'ZSL':>6} | Best @ ")
    print(f"  {'-'*6:>6} | {'-'*6:>6} | {'-'*6:>6} | {'-'*6:>6} | {'-'*6:>6} | -------")
    for alpha, r in sorted(results.items()):
        b = r['best']
        if b:
            print(f"  {alpha:>6.2f} | {b['U']:>6.2f} | {b['S']:>6.2f} | {b['H']:>6.2f} | {b['ZSL']:>6.2f} | E{b['epoch']}")


if __name__ == '__main__':
    main()
