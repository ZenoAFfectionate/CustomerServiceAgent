# -*- coding: utf-8 -*-
"""rag/knowledge_base/quality_control.py 单元测试：导入前质量检查。"""
from rag.knowledge_base.quality_control import check_blocks_quality


class TestCheckBlocksQuality:
    def test_empty_batch_fails(self):
        report = check_blocks_quality([])
        assert report["passed"] is False
        assert report["total"] == 0

    def test_healthy_batch_passes(self):
        blocks = [
            {"text": "广告限流是一种常见的风控手段，触发后曝光量会明显下降"},
            {"text": "退款需在签收后七天内申请，原路退回支付账户"},
            {"text": "物流信息可在订单详情页查看实时轨迹"},
        ]
        report = check_blocks_quality(blocks)
        assert report["passed"] is True
        assert report["warnings"] == []

    def test_high_empty_ratio_flagged(self):
        blocks = [{"text": "有效内容"}] + [{"text": ""} for _ in range(9)]
        report = check_blocks_quality(blocks)
        assert report["passed"] is False
        assert any("空文本" in w for w in report["warnings"])

    def test_high_duplicate_ratio_flagged(self):
        blocks = [{"text": "完全相同的重复内容"} for _ in range(10)]
        report = check_blocks_quality(blocks)
        assert report["passed"] is False
        assert any("重复" in w for w in report["warnings"])

    def test_short_avg_length_flagged(self):
        blocks = [{"text": "短"} for _ in range(5)]
        report = check_blocks_quality(blocks)
        assert any("过短" in w for w in report["warnings"])

    def test_html_leftover_detected(self):
        blocks = [
            {"text": "这是清洗干净的内容一二三四五六七八九十"},
            {"text": "残留了 <div>未清洗标签</div> 的内容一二三四五"},
        ]
        report = check_blocks_quality(blocks)
        assert report["html_leftover_count"] == 1
        assert any("HTML" in w for w in report["warnings"])

    def test_avg_length_and_ratios_computed_correctly(self):
        blocks = [{"text": "一二三四五六七八九十"} for _ in range(4)]
        report = check_blocks_quality(blocks)
        assert report["total"] == 4
        assert report["empty_ratio"] == 0.0
        assert report["avg_length"] == 10.0
