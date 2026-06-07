Pass 1 Decision: ACCEPTED
Pass 2 Decision: ACCEPTED
Pass 3 Decision: ACCEPTED
Overall Decision: ACCEPTED

# MOD-004 Claude 三轮审查报告

来源: `claude -p` CLI stdout。Claude CLI 未获得目标文件写入权限，因此由 Codex 按 stdout 结论回填本文件。

## Round 1：代码实现与接口契约

决策: ACCEPTED

- `VGSR.__init__` 中新增 `use_attr_patch_ot` 开关，并在开启但缺少 `class_attr` / `attr_text_embeds` 时显式报错。
- 仅当 `use_geo_attr_routing` 或 `use_attr_patch_ot` 开启时，才把 `attr_text_embeds` 传入 `CrossModalTransformer`。
- `CrossModalTransformer.forward` 复用 FAE 后 `memory` 计算 `patch_attr_sim`，形状为 `[B, N_patch, N_attr]`。
- `return_attr_patch_sim` 仅在 MOD-004 开启时为真，关闭时 `attr_patch_sim=None`。
- `compute_loss` 中新增 `loss_attr_patch_ot`，从当前类别专家属性向量取 top-K 属性并做 Sinkhorn 风格软 OT。
- loss pack 返回 `loss_attr_patch_ot`，训练日志打印 `AttrOT`。

结论: 主体实现与 review packet 的迁移设计一致，关闭路径不进入 loss，未发现接口问题。

## Round 2：配置、关闭等价性与实验隔离

决策: ACCEPTED

- 根配置默认:
  - `use_attr_patch_ot=False`
  - `lambda_attr_patch_ot=0.0`
  - `attr_patch_ot_topk=16`
  - `attr_patch_ot_temp=10.0`
  - `attr_patch_ot_iter=3`
- 实验配置打开:
  - `use_attr_patch_ot=True`
  - `lambda_attr_patch_ot=0.02`
- README 记录 baseline、seed=5 比较口径、训练命令和 `未参考官方代码`。
- Codex 自审已记录 py_compile 与 dummy forward/loss。

结论: 根配置默认关闭，实验配置局部开启，满足关闭等价性和实验隔离要求。

## Round 3：算法合理性、数值稳定性与实验风险

决策: ACCEPTED

- Sinkhorn kernel 使用 `exp((sim * temp).clamp(-20, 20))`，避免指数溢出。
- 迭代分母加 `eps=1e-8`，具备基本数值保护。
- `k_attr` 被 clamp 到合法范围。
- 训练标签使用全局类别索引取 `class_attr[labels]`，符合当前类别属性 top-K 语义。
- loss 只在 `lambda_attr_patch_ot > 0` 且 `attr_patch_sim is not None` 时进入总 loss。

实验风险:

- CUB 属性 top-K 可能引入弱相关属性噪声。
- `attr_patch_ot_iter=3` 是轻量近似，transport 不一定完全收敛。
- 均匀边缘匹配可能鼓励所有 top-K 属性都获得质量，收益需要 seed=5 正式对照验证。

## Overall

ACCEPTED

可以开始 seed=5 正式训练。
