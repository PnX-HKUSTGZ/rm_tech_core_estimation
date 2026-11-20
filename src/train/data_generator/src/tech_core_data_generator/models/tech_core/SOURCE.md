# 模型来源

- 获取仓库：<https://github.com/HKUSTGZ-ROBOMASTER-PNX/rm_tech_core_trajectory_solve>
- 固定提交：`51a52d81cc04f7e3c40e381ad652fb2595c308d9`
- 原始目录：`src/tech_core_planner/models/tech_core/`
- 获取日期：2026-11-17（Asia/Shanghai）
- 现有 `tech_core.xml` 与此提交同名文件的 SHA256 均为 `e75bbec91bd57d9e3ca3da35c36c0e4940b9fbe49932bf9dffe9a8357eac75fd`。
- OBJ 文件头注明来自 HKUST ENTERPRIZE 的 `exchange.blend`，MIT 许可、米制、E 原点、+X 朝外、+Z 向上、无 LOD；保留原始网格和文件头。
- 原作者许可全文见 `LICENSE-HKUST.txt`，取自 [RM2026-Engineer-Assembly-Algorithm](https://github.com/hkustenterprize/RM2026-Engineer-Assembly-Algorithm) 提交 `7fb7609aa6b5a8b1992d32e14bc17d24d1a15578`。

仿真运行时固定三个关节为零，隐藏碰撞代理；图像使用原始可视网格。本目录作为包数据随 `tech_core_data_generator` 一同安装。模型源自 RM2026 项目，关键点尺寸已与本仓库 RM2027 规则图核对；不等同于硬件实测标定。
