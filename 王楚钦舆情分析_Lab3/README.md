# 王楚钦微博舆情三合一决策助手（Lab 3）

Lab 3 承接 Lab 1 的案例样本和 Lab 2 的情感分析结果，提供一个“自动简报、证据约束问答、三方案建议”三合一决策台。系统先构造只含当前分析范围的 `EvidencePacket`，再生成叙述；所有统计数字与证据 ID 均来自证据包，正文和一级评论始终分开统计。

本项目用于课程中的小规模案例研究。它不代表微博总体舆情，不把描述性差异解释为因果，也不预测某项传播方案的效果。候选方案不会自动发布，最终判断、表述与执行均由人工复核决定。

## 安装与启动

以下 PowerShell 命令均在仓库根目录运行，推荐使用 Python 3.11：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\王楚钦舆情分析_Lab3\requirements.txt
```

DeepSeek API key 是可选项。优先使用环境变量：

```powershell
$env:DEEPSEEK_API_KEY = "你的 API key"
```

也可以在仓库根目录创建已被 `.gitignore` 排除的 `api.txt`：

```text
deepseek-api: 你的 API key
```

不要提交、截图或在日志中输出真实 key。启动命令为：

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\王楚钦舆情分析_Lab3\app.py
```

未配置 key 时，自动简报、预设问答和三方案建议仍可完整运行，页面会明确显示“离线”模式。在线请求超时、异常或返回不合格 JSON 时也会安全降级到确定性离线模板，不影响统计指标、证据卡和覆盖警告。

## 两分钟演示：巴黎奥运输球案例

1. 启动应用，在侧栏选择“分析范围：单场”。
2. 选择 `2024-07-31｜巴黎奥运会2024男单32强｜负`，来源选“正文与评论”，受众选“球迷”。
3. 在“自动简报”页点击“生成/刷新简报”，查看分来源指标、局限和可展开的证据记录。
4. 在“交互问答”页选择“输球是否意味着舆情全部负面？”，点击“回答预设问题”；回答会展示当前评论样本中的正、中、负构成及证据 ID。
5. 在“方案建议”页选择“回应争议”，生成三个定性候选方案。人工选择一个方案，填写复核备注，再下载 Markdown 决策简报。

这条演示链在没有 API key 时同样可用；离线回答不会补写证据包之外的事实。

## 数据流与证据边界

Lab 3 不重新采集或标注数据，启动时读取并校验以下固定输入：

- `王楚钦舆情分析_Lab1/03_说明与配置/events.json`：8 个启用赛事及胜负信息；
- `王楚钦舆情分析_Lab2/01_输出/posts_sentiment.jsonl`：45 条微博正文；
- `王楚钦舆情分析_Lab2/01_输出/comments_sentiment.jsonl`：61 条一级评论；
- `王楚钦舆情分析_Lab2/01_输出/sentiment_report.json`：分赛事、分胜负组、分来源的确定性指标；
- `王楚钦舆情分析_Lab2/02_质量报告/lab3_ingestion_check.json`：上游数据契约状态。

处理链为：加载并验证输入 → 按单场、胜组、负组或胜负对比构造证据包 → 在线生成或离线兜底 → 展示证据约束结果 → 由人工选择并导出 Markdown。导出内容包含范围、事实、观察、局限、证据记录、三个候选方案和人工备注；应用不会自动发布内容，也不会把人工备注写入数据库。

两项已知覆盖警告会持续保留：

- WTT 新加坡大满贯 2025 输球事件只有 1 条可用评论，单场结论非常稀疏；
- WTT 中国大满贯 2025 赢球事件没有可用评论，不能生成该来源的指标或引文。

## 目录

- `app.py`：Streamlit 入口与三项功能编排；
- `lab3/data_loader.py`：固定输入读取和数据契约校验；
- `lab3/evidence.py`、`lab3/models.py`：范围选择、指标摘要与可追溯证据包；
- `lab3/services.py`、`lab3/llm_client.py`：在线优先服务和无 key／异常降级；
- `lab3/offline.py`：确定性简报、五个预设问答和三方案模板；
- `lab3/export.py`：惰性 Markdown 决策记录导出；
- `lab3/ui_helpers.py`：无 Streamlit 副作用的展示辅助函数；
- `tests/`：数据契约、证据、离线、服务、界面和端到端集成测试。

## 测试

安装开发依赖后，可在仓库根目录运行全部 Lab 3 测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\王楚钦舆情分析_Lab3\requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest .\王楚钦舆情分析_Lab3\tests -q -p no:cacheprovider --basetemp C:\tmp\lab3-tests
```

仅运行核心离线集成链：

```powershell
.\.venv\Scripts\python.exe -m pytest .\王楚钦舆情分析_Lab3\tests\test_integration.py -v -p no:cacheprovider --basetemp C:\tmp\lab3-integration
```

集成测试使用真实 Lab 1/2 文件，但显式移除 API key 并把客户端指向空临时目录，因此不访问网络。

## 隐私与使用范围

输入数据沿用 Lab 1/2 的脱敏结果：用户标识已哈希，昵称已脱敏，不恢复或推断普通用户身份。原文证据只用于课程本地演示；下载前仍应由人工检查是否适合分享。

微博历史搜索和热门评论不是完整档案，样本量、赛事差异和单一模型标注都会限制结论。页面中的指标只描述当前案例，三方案只是非预测的情景比较；系统不提供自动发布、用户画像或针对个人的处置建议。
