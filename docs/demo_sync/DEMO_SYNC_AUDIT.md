# Web demo 与冻结评价口径的差异审计（**第一阶段：只读，未修改任何代码**）

**审计时间**：`2026-10-16`
**审计对象**：Streamlit 原型 `app/app.py` 及其调用的 `src/pipeline.py` 级联链路
**冻结基准**：`docs/evidence/segmentation_pipeline_v2_frozen/` 与
`docs/evidence/segmentation_eval_v2/_scripts/eval_seg_cd_test.py`（V2 正式冻结评价脚本）

> **本文件不是评价证据。** 它记录的是 **demo 实现**与冻结评价口径的差异，
> **不改动、也不重新解释**任何 V2/V3 评价结论。
> `evaluation evidence remains frozen` 与 `demo implementation synchronized after evaluation`
> 是两件事，最终交付时必须分开陈述（见文末）。

**审计阶段未修改任何文件**；改动前的代码指纹已存 `pre_change_code_sha256.json`。

> **修改阶段已完成**，见同目录 [`CHANGELOG.md`](CHANGELOG.md)：
> 其中第 1 项的 BGR→RGB 与第 7 项的高 conf 胜被做成 `frozen_compat` **开关**（默认关闭），
> 因为审计后发现 `run_pipeline` 还有三处非 demo 的历史调用方；
> 审计当时判断的"影响面"因此被修正——详见 CHANGELOG 第 2、5 节。

---

## 0. 结论速览

| # | 检查项 | 状态 |
|---|---|---|
| 1 | SAM2 前显式 BGR→RGB | ❌ **不一致** |
| 2 | detector conf | ❌ **不一致**（demo 0.25 / 冻结 0.075） |
| 3 | detector NMS IoU | ✅ 一致（0.45） |
| 4 | max_det | ✅ 一致（30） |
| 5 | ROI margin | ✅ 一致（0.10） |
| 6 | same-class mask merge | ✅ 一致（union） |
| 7 | cross-class 冲突消解 | ❌ **不一致**（demo 低 conf 胜 / 冻结 高 conf 胜） |
| 8 | 无检测时的 UI 文案 | ❌ **不一致**（"无缺陷"） |
| 9 | 无标定时的面积输出 | ✅ 一致（只报像素/占比 + "未标定"，无伪造 mm²） |

**4 项需改（1、2、7、8），5 项已一致（3、4、5、6、9）。**

---

## 1. SAM2 输入通道序

- **current behavior**：ROI 以 **BGR 原样**送入 SAM2。
  `src/pipeline.py:86` 把 `det.crop_roi()` 返回的 `roi`（`image_bgr` 的切片）直接传给
  `seg.segment_roi(sam2_predictor, roi, roi_box)`；`src/segment.py:55` 执行
  `predictor.set_image(roi_bgr)`，**全程没有 `cvtColor`**。
  （`app/app.py` 里那两处 `COLOR_BGR2RGB` 只用于 **st.image 显示**，与推理无关。）
- **frozen behavior**：显式 `cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)` 后再送 SAM2。
  `docs/evidence/segmentation_eval_v2/_scripts/eval_seg_cd_test.py:207-208`：
  `roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)` → `seg.segment_roi(sam2, roi_rgb, roi_box)`。
- **file / function**：`src/pipeline.py::run_pipeline`（第 86 行）
- **required change**：在 **`src/pipeline.py` 的 demo 路径**里显式转 RGB 后再调 `segment_roi`。
- **risk**：**高**。掩码会变（通道序影响 SAM2 的颜色先验）。
  ⚠️ **绝不能改 `src/segment.py`**：冻结评价脚本自己已经传入 RGB，
  若在 `segment_roi` 内部再加一次转换，会变成**双重转换**，
  使冻结路径无法复现 V2 的 C/D 掩码。**改动必须落在 pipeline 侧。**

## 2. detector conf

- **current behavior**：`configs/config.yaml > detect.conf = 0.25`（历史值）；
  `app/app.py:57` 的滑块初值取 `cfg.detect.get("conf", 0.25)`，用户可拖 0.05–0.95；
  `src/detect.py:35` 用 `cfg.detect.get("conf", 0.25)`。
  因此**默认演示置信度是 0.25**。
- **frozen behavior**：**0.075**，且是**内存覆盖**——
  `eval_seg_cd_test.py:68 CONF_WORKPOINT = 0.075`、第 183 行 `cfg.detect.conf = CONF_WORKPOINT`，
  **`config.yaml` 保持 0.25 不动**（这是被保护的"历史配置"）。
- **file / function**：`app/app.py::main`（第 57 行）、`configs/config.yaml`
- **required change**：让**演示默认工作点 = 0.075**（在 app 内做内存覆盖，例如调整滑块默认值）。
  **不修改 `configs/config.yaml`。**
- **risk**：**中**。conf 降到 0.075 会让**检测框显著变多**（V2 工作点 recall 低、FP 多），
  演示耗时上升、可视噪声变多；需在 UI 上说明这是**冻结评价工作点**，而非"调过的更好参数"。

## 3. detector NMS IoU

- **current behavior**：`configs/config.yaml > detect.iou = 0.45`；`src/detect.py:36` 传入 `iou=0.45`。
- **frozen behavior**：0.45。
- **file / function**：`src/detect.py::detect`
- **required change**：无。
- **risk**：无。

## 4. max_det

- **current behavior**：`config.yaml > detect.max_det = 30`；`src/detect.py:38` 传入 30。
- **frozen behavior**：30（且冻结评价记录中曾有 2 张图正好打到该上限）。
- **file / function**：`src/detect.py::detect`
- **required change**：无。
- **risk**：无。注意这个值是**上限**，演示时若框数触顶会静默丢弃后续检测，
  这与冻结口径一致，如实保留即可。

## 5. ROI margin

- **current behavior**：`src/pipeline.py:76` `det.crop_roi(image_bgr, d.xyxy, margin_ratio=0.1)`。
- **frozen behavior**：`margin_ratio=0.1`（`eval_seg_cd_test.py:201` 同值）。
- **file / function**：`src/pipeline.py::run_pipeline`
- **required change**：无。
- **risk**：无。**注意区分**：`config.yaml` 里的 `detect.box_margin`（逐类 box prompt 边距）
  是另一个量，两边都是**全 0（关闭）**，不属本项。

## 6. same-class mask merge

- **current behavior**：`src/pipeline.py:97` `full_mask[placed > 0] = int(d.cls) + 1`
  只写不清除 ⇒ 同类多次检测落在同一像素时结果相同 ⇒ **等价于 union**。
  后处理 `postprocess_mask` 内的 `keep_largest_only` 是**逐 ROI** 的，不跨 ROI 合并。
- **frozen behavior**：union（`eval_seg_cd_test.py` 同写法）。
- **file / function**：`src/pipeline.py::run_pipeline`
- **required change**：无。
- **risk**：无。

## 7. cross-class 像素冲突消解 ⚠️

- **current behavior**：`src/detect.py:50` 把检测框按 conf **降序**排序；
  `src/pipeline.py:97` 依序写入 `full_mask`，**后写的覆盖先写的** ⇒
  **低 conf 的类别覆盖高 conf 的类别 = 低 conf 胜**。
- **frozen behavior**：**高 conf 胜**。
  `eval_seg_cd_test.py:200`：`for d in sorted(dets, key=lambda x: x.conf)`（**升序**，低 conf 先写）⇒
  `pred_c[placed > 0] = v` 时高 conf 最后写 ⇒ 高 conf 胜。
  （该脚本自己的 docstring 第 9–10 行也把这一条列为与 demo 的已知分歧。）
- **file / function**：`src/pipeline.py::run_pipeline`（第 97 行的写入顺序）
- **required change**：在 **demo 路径**把叠加顺序改为 **按 conf 升序**（或在冲突像素上取 conf 最大者）。
- **risk**：**高**。多类重叠像素的归属会变 ⇒ 可视化与量化表按类面积都会变。
  ⚠️ **不要改 `src/detect.py` 的降序排序**：冻结评价脚本自己会重排，
  但消融/其它调用方依赖降序语义；**改动应局限在 demo 的叠加循环里**。

## 8. 无检测结果时的 UI 文案

- **current behavior**：`app/app.py:101` 在批量报表里写入
  `{"类别": "无缺陷", "置信度": "-", ...}`。这是**唯一一处**该措辞
  （`report/演示录屏脚本.md`、`report/研究报告.md` 中均未出现）。
- **frozen / 规定 behavior**：**统一显示「未检出目标」**，**不得写"无缺陷"**。
- **file / function**：`app/app.py::main`（第 101 行）
- **required change**：改成 `"未检出目标"`。
- **risk**：**低**（纯文案）。但这句在**演示与报告中是被明确禁止的措辞**——
  它把"模型没检出"表述成了"实际没有缺陷"，属于超出证据的断言。

## 9. 无物理标定时的面积输出

- **current behavior**：`src/quantify.py:25-29` `pixels_to_area_mm2()` 在
  `mm_per_pixel is None` 时**返回 None**；`to_table_rows()`（第 68/74 行）在 None 时输出
  **「未标定」**；`app/app.py:98` 的批量报表同样输出「未标定」；
  `app/app.py:55` 的标定输入框留空即 `None`，help 写明"留空则只输出像素面积与占比"。
  ⇒ **没有伪造 mm²**，且明确标了「未标定」。
- **frozen / 规定 behavior**：吻合。
- **file / function**：`src/quantify.py`、`app/app.py`
- **required change**：**无需功能改动**。可选：把表头「真实面积(mm²)」在未标定时
  改成「真实面积(mm²，未标定)」之类更醒目的写法——**属于可选项，不作为必改项**。
- **risk**：无。

---

## 12. 改动时的硬约束（先写下来，避免越界）

1. **不得改 `configs/config.yaml`**：`detect.conf: 0.25` 是 V2 的历史值，受保护。
   演示的 0.075 只能**内存覆盖**。
2. **不得改 `src/segment.py` 的通道序行为**（冻结评价依赖它接收已转换好的 RGB）。
3. **不得改 `src/detect.py` 的排序语义**（其它调用方依赖降序）。
4. 改动只落在 **`src/pipeline.py`（新增可选的兼容开关，默认不改变原行为）** 与
   **`app/app.py`**（演示参数与文案）。
5. **不跑任何冻结 test 集**，不生成任何新的科研性能指标；改完只做**功能 smoke test**。
6. 保留**改动前后代码 SHA256** 与 **change log**。

## 13. 交付时的两句话（必须分开写，不得混为一谈）

```
evaluation evidence remains frozen
  —— V2/V3 的全部正式指标仍来自冻结件（final_test_v2 / v3_test_eval），
     本次 demo 同步**没有**重跑评价、**没有**产生新的性能数字。

demo implementation synchronized after evaluation
  —— 为现场演示，Web 原型的处理链路已对齐 V2 冻结评价口径
     （通道序 / conf 0.075 / 跨类高 conf 胜 / UI 文案）；
     这只是**实现一致化**，**不是**"重新评价后模型变好了"。
```

**不得**把同步后的 demo 结果当作重新评价后的模型性能。
