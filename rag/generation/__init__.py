# -*- coding: utf-8 -*-
"""生成能力子模块（Generation）：Prompt 模板、上下文组装、LLM 生成、引用构建、
幻觉控制。

    - prompt_template.py       系统提示词 / 用户提示词拼装
    - context_assembly.py      检索上下文 → 送入 LLM 的文本（字符预算控制）
    - llm_generation.py        vLLM 生成 / 本地抽取式兜底（对外统一入口 generate_answer）
    - citation.py               引用列表构建
    - hallucination_control.py 生成结果事后校验（引用有效性 + 关联度评分）
"""
