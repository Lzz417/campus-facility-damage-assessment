# Demo sync CHANGELOG

```
Scientific evaluation:
FROZEN / UNCHANGED

Demo implementation:
SYNCHRONIZED AFTER EVALUATION

New performance evidence:
NONE
```

> These engineering changes were **not** used to regenerate or replace any reported
> V2/V3 evaluation result.

**日期**：`2026-10-16` ｜ **范围**：Web demo pipeline 与 V2 冻结评价口径的工程同步
**未做**：不新增实验、不改冻结科研证据、不跑任何冻结 test 集、不生成新的性能指标。

---

## 1. 改动清单（4 项）

1. **Explicit BGR → RGB conversion before SAM2.**
   `src/pipeline.py::run_pipeline` 在调用 `seg.segment_roi()` 前执行
   `cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)`（仅 `frozen_compat=True` 时）。
   **未修改 `src/segment.py`**：冻结评价脚本 `eval_seg_cd_test.py` 已自行传 RGB，
   若在 `segment_roi` 内再转换会造成**双重转换**、破坏 V2 C/D 掩码可复现性。

2. **Demo runtime detector confidence default changed to frozen V2 operating point 0.075
   without modifying historical `config.yaml`.**
   `app/app.py` 新增模块常量 `FROZEN_CONF = 0.075`，滑块默认值改为它；
   偏离时显示「⚠️ 已偏离冻结工作点…此时**不再是**冻结工作点演示」。
   **`configs/config.yaml` 的 `detect.conf: 0.25` 一字未改**（受保护历史值），
   0.075 只在 demo 运行时内存覆盖。

3. **Cross-class overlap changed to highest-confidence-wins.**
   `src/pipeline.py` 在 `frozen_compat=True` 时按 conf **升序**遍历（低 conf 先写、高 conf 后写）
   ⇒ 重叠像素归**高 conf** 类；同类别仍为 union。
   仅对**遍历副本**排序，**未修改 `src/detect.py` 的降序返回语义**。

4. **「无缺陷」 changed to 「未检出目标」.**
   `app/app.py` 无检测分支的报表文案改为「未检出目标」。
   不得把"模型未检出"解释成"实际不存在缺陷"。

**改动文件**：`src/pipeline.py`、`app/app.py`（`git diff --shortstat`：**2 files changed, 48 insertions(+), 6 deletions(-)**）。
**未改动**：`src/segment.py`、`src/detect.py`、`src/postprocess.py`、`src/quantify.py`、
`configs/config.yaml`、`start_system.py`（SHA256 前后一致，见第 4 节）。

## 2. 关键工程决定：为什么加了 `frozen_compat` 开关

审计时发现 **`run_pipeline` 不是 demo 专用**，另有三处历史调用方：

```
experiments/ablation.py:111
scripts/eval_box_margin_effect.py:63
scripts/eval_branch_d.py:44
scripts/make_demo_assets.py:58
```

若直接改 `run_pipeline` 的默认行为，这些脚本**重跑时会得到与已记录的
`experiments/ablation*.csv` 不同的结果**（静默不可复现）。

因此改动做成**显式开关 `frozen_compat: bool = False`**：

- **默认 `False` = 完全保持历史行为**（BGR 原样送 SAM2；跨类冲突低 conf 胜）⇒ 历史调用方**零影响**；
- **`True` 仅由 `app/app.py` 传入** ⇒ 只有 demo 走冻结口径。

这一条有专门的反向回归测试（smoke test `3c`）守着，而不是靠"我觉得没影响"。

**冻结评价不受影响**：`eval_seg_cd_test.py` **不调用** `run_pipeline`
（它自己重实现了放置逻辑，这正是 V2 记录在案的原因），故本次改动**不触碰任何评价路径**。

## 3. Smoke test 结果（9/9 通过）

`docs/demo_sync/smoke_test.py` → `smoke_test_result.json`。**不跑冻结 test、不算科研指标**。

| # | 项 | 结果 |
|---|---|---|
| 1 | 端到端跑通（YOLO→ROI→SAM2→输出，非冻结 test 图） | PASS：检测数 1，掩码 shape (720,1280) |
| 2 | 送入 SAM2 的确实是 RGB | PASS：1/1 ROI 与 `cvtColor(BGR2RGB)` 逐像素相同，且与 BGR 有差异 |
| 3 | 重叠区归**高 conf** 类（demo 路径） | PASS：重叠像素类别值=2（cls1 crack，高 conf） |
| 3b | 非重叠区未受影响 | PASS：两 ROI 之外为背景；仅高 conf 覆盖处为该类 |
| **3c** | **默认路径行为未变（最小影响）** | PASS：默认下重叠像素=3（cls2，低 conf 胜 = 历史行为），且**送 SAM2 的是 BGR 原样**（与 BGR 逐像素相同=True / 与 RGB 相同=False） |
| 4 | 无检测文案 | PASS：app.py 含「未检出目标」；**UI 代码中违规 0 处** |
| 5 | 无标定只报像素/占比+「未标定」 | PASS：`area_mm2=None`、表格单元格「未标定」、像素/占比正常；有标定时 mm²=25.00 |
| 6 | 受保护文件 SHA256 未变 | PASS：`configs/config.yaml`／`src/segment.py`／`src/detect.py` 全部未变 |
| 7 | 权重与评价证据零写入 | PASS：评价证据目录被写 0 个；模型权重文件被写 0 个；两个已发布权重 SHA256 全部未变 |

**测试图的来源**：`data/raw/MBDD/MBDD2025/JPEGImages/Hefei1000.jpg`，
已代码内校验其**不在**冻结 split_v2 test 的 1,179 张里（属 train）。

## 4. pre / post SHA256

| 文件 | pre | post | 变化 |
|---|---|---|---|
| `app/app.py` | `9f50476b9f28c9a1…` | `cde0cf8b42e2…` | **YES（本次改动）** |
| `src/pipeline.py` | `fba9ae8eb6c999fa…` | `06d593a788cd…` | **YES（本次改动）** |
| `src/segment.py` | `ad821c07f298d4d6…` | 同左 | **no** ✅ |
| `src/detect.py` | `67233097ef3c7cf9…` | 同左 | **no** ✅ |
| `src/postprocess.py` | `97cb4599c2da0c78…` | 同左 | no |
| `src/quantify.py` | `3fce0220d1b55992…` | 同左 | no |
| `configs/config.yaml` | `bf58f11a95d3f05f…` | 同左 | **no** ✅ |
| `start_system.py` | `8781a91f1a72a596…` | 同左 | no |

完整清单：`pre_change_code_sha256.json` / `post_change_code_sha256.json`。
实际 diff：`CHANGELOG_diff.patch`。

## 5. 本次工作中发现并修正的问题（如实记录）

| # | 问题 | 性质 | 处理 |
|---|---|---|---|
| A | 审计阶段未发现 `run_pipeline` 有非 demo 调用方 | **影响面判断失误**（不是代码错） | 改为 `frozen_compat` 开关，默认保持历史行为；补 3c 反向回归 |
| B | 编辑 docstring 时把字符串提前闭合，文档剩余部分变成非法代码 | **我自己的编辑失误** | 立即修正；`py_compile` 通过后才继续 |
| C | smoke test 的 3c 用了 `np.array_equal(...) is False` | **测试自身缺陷**（`np.bool_` 不是 Python `False` 单例）→ 断言恒假、等于没测 | 改为直接与 BGR 切片比对 |
| D | 3c 的合成图是均匀灰 ⇒ BGR==RGB，通道序断言无法判别 | **测试设计缺陷** | 改用有色合成图（三通道不同） |
| E | smoke test 初次把注释里的字面量也算成违规 | **检查器口径过宽** | 限定为"UI 代码中的文案"，其余语义（数据集清洗）单独列出并保留 |
| F | 上一轮声称"报告文件不存在" | **我的检索失误** | 报告实际在 `report/研究报告.md`；更正 previous file-discovery statement，**不影响任何科研结果** |
| G | 发布审计中把 `train/train_yolo_v2.py` 判为"0 绝对路径" | **bash grep 反斜杠转义写错** ⇒ 误判；该文件第 27 行实为 `ROOT = Path(r"D:\zl	iaobei")` | 已改为 `Path(__file__).resolve().parent.parent`；并改用 Python 脚本做扫描 |
| H | 用正则扫绝对路径时把 `https://` 与 Python 的 `f:` / `s:` 字典语法判为命中 | **我的正则过宽**（盘符后的分隔符写成了可选） | 改为「盘符 + 必须跟分隔符 + 前导非字母数字」，重扫后为 0 |
| I | smoke test 的 `test_7` 把 `weights/` 下任何文件都判为违规 | **检查器过宽**：把故意新增的 `weights/README.md` 误报为风险 | 改为**只判模型权重文件**并**显式核对两个已发布权重的 SHA256** |

**没有任何意外写入**：第 7 项检查确认改动窗口内 `docs/evidence`、`experiments`、
`weights`、`data/splits*` 均无文件被写。

## 6. 交付边界的最终表述（不得混为一谈）

```
evaluation evidence remains frozen
  —— V2/V3 的全部正式指标仍来自冻结件；本次同步没有重跑评价、没有产生新性能数字。

demo implementation synchronized after evaluation
  —— Web 原型的处理链路已对齐 V2 冻结评价口径（通道序 / conf 0.075 /
     跨类高 conf 胜 / UI 文案）；这只是实现一致化。
```

**不得**把同步后的 demo 当作"重新评价后模型变好了"。
**不得**把 V3 detector 换进完整 YOLO→SAM2 演示链并称为「V3 系统」——
V3 未做像素级分割验证，完整分割演示仍按**已有 V2 分割证据**的对应关系处理。
