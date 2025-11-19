你现在在一个已有的 Python 项目仓库 SoftTeacherDrift 里工作。

该仓库中有一个文档入口：docs/PROJECT_OVERVIEW.md 和 docs/INDEX.md。
请先阅读这两个文件，理解项目目标、代码结构和“给代码助手的规则”。

之后你要遵守以下约定：
1. 在修改任何模块之前，优先根据 docs/INDEX.md 里的索引，找到对应的模块文档（docs/modules/*.md），先阅读相关说明。
2. 修改或新增代码时，要保持接口和约定与文档一致；如果需要改变约定，请同时更新对应模块的 md 文件。
3. 每次完成一组改动后：
   - 在 changelog/CHANGELOG.md 的末尾新增一条记录，格式类似：
     - `YYYY-MM-DD  [assistant]  修改 summary：{一句话说明主要改动}`
     - `影响文件：models/..., training/..., etc.`
   - 如涉及设计上的变更（例如增加新的 drift 指标、改变日志字段），要在对应 docs/modules/*.md 中更新“当前实现状态”和“使用方法”小节。
4. 文档更新要简洁，以 bullet list 和小标题为主，避免一大段没有结构的文字。

接下来，我会给你具体的任务（例如增加一个新的 drift metric、修改训练 loop 的日志格式等），你在实现代码的同时，要按照上述规则同步更新文档。
