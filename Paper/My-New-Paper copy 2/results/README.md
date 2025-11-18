# 结果目录说明

## 目录内容
存放所有实验的原始输出结果，包括：

- **CSV文件**: 实验数据记录
- **日志文件**: 训练和测试日志
- **配置文件**: 实验配置参数
- **中间结果**: 实验过程中的输出

## 文件命名规范
建议采用以下命名格式：
- `YYYY-MM-DD_experiment_name_results.csv`
- `YYYY-MM-DD_experiment_name.log`
- `config_experiment_name.yaml`

## 注意事项
- 本目录内容不会被 Git 跟踪（已在 .gitignore 中配置）
- 重要结果应及时备份
- 建议定期清理临时文件