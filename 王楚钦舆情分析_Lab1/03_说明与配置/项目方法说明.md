# Lab 1：王楚钦赢球/输球后微博舆情数据

## 研究问题

比较王楚钦在重要单打比赛获胜和失利后 24 小时内的微博舆情。项目只描述所选比赛案例，
不把结果推广为王楚钦所有比赛的普遍规律。

## 最终事件样本

| 结果 | 日期 | 赛事 | 对手 | 轮次 |
|---|---|---|---|---|
| 胜 | 2025-03-16 | WTT 重庆冠军赛 | 林诗栋 | 决赛 |
| 胜 | 2025-05-25 | 多哈世乒赛 | 卡尔德拉诺 | 决赛 |
| 胜 | 2025-07-14 | WTT 美国大满贯 | 张本智和 | 决赛 |
| 胜 | 2025-10-05 | WTT 中国大满贯 | 费利克斯·勒布伦 | 决赛 |
| 负 | 2024-07-31 | 巴黎奥运会 | 莫雷加德 | 32 强 |
| 负 | 2025-02-08 | WTT 新加坡大满贯 | 梁靖崑 | 半决赛 |
| 负 | 2025-04-19 | 澳门世界杯 | 卡尔德拉诺 | 半决赛 |
| 负 | 2025-08-11 | WTT 横滨冠军赛 | 张本智和 | 决赛 |

赛果通过 WTT、新华社、央视等公开报道交叉核验。例如：[重庆夺冠](https://www.worldtabletennis.com/description?artId=4732)、
[多哈夺冠](https://www.worldtabletennis.com/description?artId=4983)、
[美国夺冠](https://m.gmw.cn/2025-07/14/content_1304082536.htm)、
[中国大满贯夺冠](https://www.worldtabletennis.com/description?artId=5479)、
[巴黎奥运失利](https://www.news.cn/sports/20240731/27be3c810f3241ddb1f3b44a0cdc0dde/c.html)、
[新加坡失利](https://www.news.cn/sports/20250208/b6788ec3f25145d1b810f8a3ff939abd/c.html)、
[横滨失利](https://www.worldtabletennis.com/description?artId=5262)。

`events.json` 是事件、时间窗口和关键词的唯一配置来源。两场探索性胜局因历史搜索召回过低，
以 `active: false` 保留审计记录，但不进入最终比较。

## 关键词 taxonomy

- 人物：王楚钦、对手姓名；
- 赛事：赛事名称、城市、轮次；
- 时间：比赛日期；
- 结果：明确比分、夺冠、晋级、无缘决赛等。

日期和比分用于定位事件，不作为情绪标签。清洗过程不会因为文本中出现正面或负面词而删除记录。

## 清洗规则

1. 仅保留配置的赛后 24 小时窗口；
2. 正文必须包含“王楚钦”，并命中对手别名或赛事词；
3. 清理 HTML、链接和多余空白；
4. 在单场比赛内按微博 ID 和完全相同的规范化正文去重；
5. 用户仅保留项目已有的匿名哈希；
6. 对明显广告文本添加标记，不进行主观情绪筛选。

## 输出文件

- `output/wang_chuqin_weibo_clean.jsonl`、`.csv`：完整清洗数据；
- `output/wang_chuqin_weibo_balanced.jsonl`、`.csv`：每场最多保留赛后最早 6 条，推荐交给 Lab 2；
- `output/quality_report.json`：每场原始量、窗口外数量、去重数量和最终数量。
- `output/wang_chuqin_weibo_comments_24h.jsonl`、`.csv`：45 条均衡微博下、落在各场赛后 24 小时内的一级评论；
- `output/wang_chuqin_weibo_comments_analysis_ready.jsonl`、`.csv`：在上述评论中排除明显泛互动文本后的推荐分析集；
- `output/wang_chuqin_weibo_comments_balanced.jsonl`、`.csv`：在分析集上每场最多取 10 条，推荐用于胜负比较；
- `output/comments_quality_report.json`：评论的时间窗、胜负、事件和噪声统计。

评论和微博正文使用相同的 `event_id`、`match_result` 与时间窗口，但通过 `content_type` 分开分析。
每条锚点最多抓取 10 条一级热门评论，不抓二级回复。移动端接口返回的是热门流，不是完整的时间顺序档案；
因此评论样本代表这 45 个评论区中可获取的热门评论，不能代表微博全部用户。

重新清洗：

```powershell
.\.venv\Scripts\python.exe lab1\clean_lab1.py `
  --raw lab1\raw_batch `
  --raw lab1\raw_supplement `
  --raw lab1\raw_replacement `
  --raw lab1\pilot_yokohama_dated `
  --output lab1\output
```

## 局限

微博移动端历史搜索不是完整档案：结果受当前排序和索引保留影响。因此，本数据适合课程中的小规模案例比较，
不能用于估计微博总体舆情比例。不同赛事级别和轮次也会影响讨论强度，后续分析应按 `event_id` 先分场统计，
再汇总胜负组，避免单场热点主导结论。
