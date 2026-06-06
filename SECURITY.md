# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Supported       |

## 安全注意事项

- **API 密钥安全**：请通过 `.env` 文件配置 API 密钥，不要将其提交到代码仓库
- **数据传输**：敏感数据请优先使用本地 LLM 部署，避免数据外泄
- **输入过滤**：确保输入数据的编码为 UTF-8，避免注入攻击

## Reporting a Vulnerability

如发现安全漏洞，请通过 [GitHub Issues](https://github.com/huajielong/SensFinder/issues) 报告，或发送邮件至项目维护者。

请勿在公开渠道披露未修复的安全漏洞。
