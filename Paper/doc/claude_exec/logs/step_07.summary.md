完成。

## Schema扫描结果
- ✅ **ok_runs**: 11
- ✅ **bad_runs**: 0

所有扫描到的 run_meta.yaml 文件均通过验证，无需进入修复工单。

## 命令输出摘要
- 扫描路径：`save/` 和 `outputs/`
- 验证函数：`uxfd.io.validate.validate_run_dir`
- 结果：11个运行目录全部通过schema验证
- 失败列表：无

验收标准达成：`bad_runs == 0`，所有schema文件均符合规范。
