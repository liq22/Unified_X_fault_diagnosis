import os
import csv
import yaml

def find_versions(logs_path):
    """返回 logs 目录下所有 version_x 文件夹的绝对路径"""
    return [os.path.join(logs_path, d) for d in os.listdir(logs_path)
            if d.startswith('version_') and os.path.isdir(os.path.join(logs_path, d))]

def get_target(hparams_path):
    """读取 hparams.yaml，返回 target 字段"""
    try:
        with open(hparams_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('target', '')
    except Exception:
        return ''

def get_last_test_acc(metrics_path):
    """读取 metrics.csv，返回最后一行的 test_acc"""
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in reversed(rows):
                if row.get('test_acc'):
                    return row['test_acc']
        return ''
    except Exception:
        return ''

def main():
    save_root = os.path.join('script', 'save')
    result = []
    for task in os.listdir(save_root):
        task_path = os.path.join(save_root, task)
        if not os.path.isdir(task_path):
            continue
        for model in os.listdir(task_path):
            model_path = os.path.join(task_path, model)
            if not os.path.isdir(model_path):
                continue
            for exp in os.listdir(model_path):
                exp_path = os.path.join(model_path, exp)
                logs_path = os.path.join(exp_path, 'logs')
                if not os.path.isdir(logs_path):
                    continue
                for version_path in find_versions(logs_path):
                    hparams_path = os.path.join(version_path, 'hparams.yaml')
                    metrics_path = os.path.join(version_path, 'metrics.csv')
                    target = get_target(hparams_path)
                    test_acc = get_last_test_acc(metrics_path)
                    result.append({
                        'task': task,
                        'model': model,
                        'exp': exp,
                        'version': os.path.basename(version_path),
                        'target': target,
                        'test_acc': test_acc
                    })

    # 写入 result_acc.csv
    out_path = os.path.join(save_root, 'result_acc.csv')
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'model', 'exp', 'version', 'target', 'test_acc'])
        writer.writeheader()
        writer.writerows(result)
    print(f'已保存到 {out_path}，共 {len(result)} 条记录。')

if __name__ == '__main__':
    main()