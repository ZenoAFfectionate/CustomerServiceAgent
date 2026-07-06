# -*- coding: utf-8 -*-
"""model/trainer/reranker_dpo.py 单元测试：DPO 损失计算 + 设备相关的
dtype 选择逻辑（不依赖 GPU/真实模型权重，仅测试纯函数/常量部分）。

注：`reranker_dpo.py` 模块级 import 了 `wandb`（当前开发/测试环境未安装），
但该依赖仅用于 `main()` 训练流程中的实验跟踪，`compute_dpo_loss` 等纯函数
不会用到；这里用最小 stub 占位以便可测试模块导入，避免因缺少与本测试无关
的可选训练依赖而无法验证核心损失函数的数学正确性。
"""
import importlib.machinery
import sys
import types

import pytest

torch = pytest.importorskip("torch")

if "wandb" not in sys.modules:
    _wandb_stub = types.ModuleType("wandb")
    # 需要设置合法的 __spec__，否则 `accelerate`（transformers 的间接依赖）
    # 用 `importlib.util.find_spec("wandb")` 探测可用性时会因 stub 模块没有
    # __spec__ 而抛 ValueError（而非仅返回 None）。
    _wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    _wandb_stub.init = lambda *a, **kw: None
    _wandb_stub.log = lambda *a, **kw: None
    sys.modules["wandb"] = _wandb_stub

from model.trainer.reranker_dpo import compute_dpo_loss, MODEL_DTYPE, device  # noqa: E402


class TestModelDtypeSelection:
    """回归测试：修复审查报告 M4——fp16 权重在 CPU 上做前向/反向传播会因
    多数 PyTorch 版本不支持而报错，DPO 训练在无 GPU 环境下完全无法运行。
    现应按设备类型选择 dtype：仅 CUDA 可用时使用 fp16，CPU 环境回退 fp32。
    """

    def test_dtype_matches_device_availability(self):
        if device.type == "cuda":
            assert MODEL_DTYPE == torch.float16
        else:
            assert MODEL_DTYPE == torch.float32


class TestComputeDpoLoss:
    """回归测试：修复审查报告 M4——DPO loss 此前直接对回归头原始分数求差值，
    未 log_sigmoid 归一化为 log-prob。现验证其数学行为符合预期。"""

    def test_loss_is_lower_when_policy_prefers_chosen_more_than_reference(self):
        """策略模型比参考模型更强烈地偏好 chosen 时，loss 应更低（偏好被正确强化）。"""
        chosen = torch.tensor([2.0, 2.0])
        rejected = torch.tensor([0.0, 0.0])
        ref_chosen = torch.tensor([0.5, 0.5])
        ref_rejected = torch.tensor([0.5, 0.5])

        strong_pref_loss, strong_pref_acc = compute_dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)

        # 参考模型对两者打分相同（无偏好）时，策略模型的偏好差异全部计入 loss；
        # 弱偏好场景：策略模型 chosen/rejected 分差更小
        weak_chosen = torch.tensor([1.0, 1.0])
        weak_pref_loss, weak_pref_acc = compute_dpo_loss(weak_chosen, rejected, ref_chosen, ref_rejected, beta=0.1)

        assert strong_pref_loss.item() < weak_pref_loss.item()
        assert strong_pref_acc.item() == 1.0
        assert weak_pref_acc.item() == 1.0

    def test_accuracy_reflects_raw_score_comparison(self):
        chosen = torch.tensor([1.0, -1.0])
        rejected = torch.tensor([0.5, 0.5])
        ref = torch.zeros(2)

        _, acc = compute_dpo_loss(chosen, rejected, ref, ref, beta=0.1)
        # 第一条 chosen(1.0) > rejected(0.5) 为真；第二条 chosen(-1.0) > rejected(0.5) 为假
        assert acc.item() == pytest.approx(0.5)

    def test_loss_uses_log_sigmoid_not_raw_score_difference(self):
        """核心回归点：验证 loss 计算确实经过 log_sigmoid 变换，而非直接对
        原始分数求差（二者在大分值区间的数值行为显著不同，因 sigmoid 饱和）。"""
        import torch.nn.functional as F

        chosen = torch.tensor([10.0])
        rejected = torch.tensor([-10.0])
        ref_chosen = torch.tensor([0.0])
        ref_rejected = torch.tensor([0.0])
        beta = 0.1

        loss, _ = compute_dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=beta)

        # 手动按"修复后"的 log-sigmoid 公式复算，应与实现一致
        policy_log_ratio = F.logsigmoid(chosen) - F.logsigmoid(rejected)
        reference_log_ratio = F.logsigmoid(ref_chosen) - F.logsigmoid(ref_rejected)
        expected_logits = beta * (policy_log_ratio - reference_log_ratio)
        expected_loss = -F.logsigmoid(expected_logits).mean()
        assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-5)

        # 若仍是"修复前"直接用原始分数差值的实现，loss 会明显不同（大分值下
        # beta*(20-0)=2.0，-logsigmoid(2.0)≈0.127，而 log-sigmoid 版本因两端
        # 均已趋近饱和值，loss 应显著更小）
        raw_diff_logits = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
        raw_diff_loss = -F.logsigmoid(raw_diff_logits).mean()
        assert loss.item() != pytest.approx(raw_diff_loss.item(), abs=1e-5)
