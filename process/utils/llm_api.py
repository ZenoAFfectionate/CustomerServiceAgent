# -*- coding: utf-8 -*-
"""
LLM 摘要生成与多轮问答重写模块。

封装 ChatGLM 本地模型和 vLLM HTTP 服务两种调用方式，提供：
1. 文档块摘要生成（同步 / 异步）
2. 问题生成
3. 多轮对话 Query 重写（vLLM / Ollama / ChatGLM）
4. 向量嵌入获取

核心函数：
    - infer_chunk_category:             根据 URL 分类文档内容
    - generate_summary_vllm:            同步 vLLM 摘要
    - generate_summary_vllm_async:      异步 vLLM 摘要（多实例负载均衡）
    - generate_summary_ChatGLM:         ChatGLM 本地摘要
    - generate_question_ChatGLM:        ChatGLM 问题生成
    - rewrite_query_vllm_async:         异步 vLLM Query 重写
    - rewrite_query_ChatGLM:            ChatGLM Query 重写
    - get_embedding_from_vllm:          vLLM 嵌入向量
"""

import requests
import time
import asyncio
import random
from typing import List, Dict, Optional, Set

# [Optimized] aiohttp 仅用于异步函数（generate_summary_vllm_async / rewrite_query_vllm_async），
# 未安装时降级为 None，同步函数不受影响（与 config.py 保持一致的容错策略）。
try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore

from utils.config import CONFIG, sem, logger, OLLAMA_API_URL


# ======================== 常量 ========================

VLLM_TIMEOUT = CONFIG.get("vllm_timeout", 60)
VLLM_SERVERS = CONFIG.get("vllm_api_servers", [])

# Query 重写过滤词（寒暄、灌水、指令等无需重写的短句）
REWRITE_BANNED_PHRASES = frozenset([
    "你好", "您好", "hi", "hello", "哈喽", "在吗", "喂", "请问在吗", "有人吗", "hello？",
    "你是谁", "你是人吗", "你是机器人吗", "你是AI吗", "你叫什么", "你是客服吗",
    "你是智能助手吗", "你是人工的吗", "你能听懂我说话吗",
    "呵呵", "哈哈", "嗯", "哼", "额", "好吧", "无语", "。。。", "...", "===",
    "你猜", "随便", "看你咋说",
    "测试", "test", "just testing", "随便问问", "这是个测试", "debug", "看看你怎么回答",
    "今天是几号", "时间", "天气", "北京天气", "今天天气", "讲个笑话", "背首诗",
    "给我唱首歌", "来段rap",
    "重启一下", "清除缓存", "退出系统", "保存文件", "打开浏览器", "运行代码",
    "执行脚本", "回答问题",
    "你扮演谁", "假设你是", "你是人类", "如果你是我", "从你的角度看", "你作为一个AI",
    "存在的意义是什么", "人生的意义", "什么是真实", "你怎么看这个世界", "你觉得我是谁",
    "商品", "服务", "平台", "抖音", "小红书", "视频", "规则", "政策", "报表",
])

REWRITE_SYSTEM_PROMPT = (
    "你是一个专业的问题重写模块，专门用于多轮对话场景下的指代补全与语义还原任务。\n"
    "请严格按照以下规则执行：\n\n"
    "【任务目标】\n"
    "1. 你的目标是准确解析用户真实意图，从历史中发掘指代和意图，"
    "使得这个独立问题尽量完整，尽可能包含所有信息关键词,"
    "特别是要补充关键的动机，指代和场景"
    "（润色后的问题至少涉及：用户面对的前置情况，什么限制，什么疑问等）。\n"
    "2. 仅当历史信息能够提供明确上下文时才进行补全，否则保持当前问题不变。\n"
    "3. 对于明显不需要补全的问题如：语气词：\"呵呵\"，命令：\"关机\"，"
    "无关内容：\"人生的意义\"等，返回原句\n"
    "4. 不进行任何无关发挥、扩写、润色、修辞性描述、解释、总结、感情色彩。\n"
    "5. 不编造任何不存在的假设背景或新信息。\n"
    "6. 输出格式必须严格遵循：仅输出最终重写结果文本，不包含任何前缀、提示词、"
    "说明性文字或换行符。重写的疑问句以：\"我想知道\"开头，非疑问句你自行适配\n"
    "7. 当无需重写时，直接输出原问题。\n\n"
    "【重写示例】\n"
    "历史对话：\n用户：小红鞋什么时候有货？\n系统：请问您指的是哪款小红鞋？\n"
    "用户：就是上次缺货那款\n当前问题：就是上次缺货那款\n"
    "重写结果：我想知道上次缺货的那款小红鞋什么时候有货？\n\n"
    "历史对话：\n用户：我们前几天账号被限流了，不知道什么原因\n"
    "系统：限流可能是因为素材违规或账户表现不佳。\n"
    "用户：那我们还有其他计划，能投放吗？\n"
    "当前问题：那我们还有其他计划，能投放吗？\n"
    "重写结果：我想知道当账号处于限流状态时，是否可以继续开启新的广告投放计划？\n\n"
    "历史对话：\n用户：你是谁？\n当前问题：你是谁？\n重写结果：你是谁？\n\n"
    "历史对话：\n用户：开心\n当前问题：开心\n重写结果：开心\n\n"
    "历史对话：\n用户：我要买它\n系统：请问您指的是哪款商品？\n"
    "用户：之前推荐的那双\n当前问题：之前推荐的那双\n"
    "重写结果：我想要买你之前推荐的那双鞋。\n\n"
    "注意：严格按照以上风格工作，禁止任何额外输出。"
)


# ======================== 文档分类 ========================

# 分类关键词映射
_CATEGORY_KEYWORDS = [
    (["规则", "制度", "法律", "审核"], "规则类"),
    (["使用", "指南", "帮助", "操作", "功能"], "操作类"),
    (["生态", "角色", "策略", "推广", "平台信息"], "信息类"),
]


def infer_chunk_category(page_url: str) -> str:
    """根据 URL 路径分类文档内容（规则/操作/信息/泛用）。"""
    for keywords, category in _CATEGORY_KEYWORDS:
        if any(k in page_url for k in keywords):
            return category
    return "泛用类"


# ======================== 摘要 prompt 构建 ========================

def _build_summary_prompt(text: str, page_url: str, max_new_tokens: int) -> str:
    """构建摘要生成的提示词（vLLM 和 ChatGLM 共用）。"""
    category = infer_chunk_category(page_url)
    return (
        f"你正在处理一篇电商平台的知识内容，属于「{category}」类。\n"
        f"请你根据下方内容提炼其主要信息，要求如下：\n"
        f"1. 概括要点，不要重复原文原句；\n"
        f"2. 总长度不超过{max_new_tokens}字，使用简体中文；\n"
        f"3. 输出格式为完整一句话。\n"
        f"📂 来源路径：{page_url}\n"
        f"📄 内容：\n{text}"
    )


# ======================== vLLM 摘要 ========================

def generate_summary_vllm(text: str, page_url: str, max_new_tokens: int = 150, model: str = "glm") -> str:
    """同步调用 vLLM 生成摘要。短文本直接截断返回。"""
    if len(text) < max_new_tokens * 2:
        logger.debug("⚠️ 文本长度不足，使用原文本")
        return text[:max_new_tokens]

    text = text.strip().replace("\x00", "")
    prompt = _build_summary_prompt(text, page_url, max_new_tokens)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8,
    }

    api_url = CONFIG.get("vllm_api_url", "http://localhost:8011/v1/chat/completions")

    try:
        start = time.time()
        response = requests.post(api_url, json=payload, timeout=VLLM_TIMEOUT)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        logger.debug(f"✅ vLLM摘要成功 (耗时 {time.time() - start:.2f}s)")
        return result or text[:max_new_tokens]
    except Exception as e:
        logger.warning(f"⚠️ vLLM 摘要生成失败: {e}，fallback 到截断文本")
        return text[:max_new_tokens]


async def generate_summary_vllm_async(text: str, page_url: str, model: str = "glm", max_new_tokens: int = 150) -> str:
    """异步调用 vLLM 生成摘要，支持多实例负载均衡与失败重试。"""
    text = text.strip().replace("\x00", "")
    fallback_summary = text[:max_new_tokens]

    prompt = _build_summary_prompt(text, page_url, max_new_tokens)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8,
    }

    try:
        async with sem:
            start = time.time()
            logger.debug(f"📩 vLLM 异步摘要请求: {page_url} {text[:64]}")
            result = await call_vllm_with_retry_weighted(payload, timeout=VLLM_TIMEOUT)
            summary = result["choices"][0]["message"]["content"].strip()
            logger.debug(f"✅ vLLM 异步摘要成功 (耗时 {time.time() - start:.2f}s)")
            return summary or fallback_summary
    except Exception as e:
        logger.error(f"⚠️ vLLM 异步摘要失败: {e}，返回截断文本")
        return fallback_summary


# ======================== ChatGLM 推理共享 ========================

def _chatglm_generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    """ChatGLM 模型推理的共享函数。

    封装 apply_chat_template → model.generate → decode 三步流程。
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
        )

    # 裁剪掉 prompt 部分，只保留生成内容
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return response


def generate_summary_ChatGLM(text: str, page_url: str, model, tokenizer, max_new_tokens: int = 150) -> str:
    """使用 ChatGLM 本地模型生成摘要。"""
    if len(text) < max_new_tokens * 2:
        logger.debug("⚠️ 文本长度不足，使用原文本")
        return text[:max_new_tokens]

    text = text.strip().replace("\x00", "")
    prompt = _build_summary_prompt(text, page_url, max_new_tokens)

    try:
        response = _chatglm_generate(model, tokenizer, prompt, max_new_tokens, 0.4, 0.8)
        return response if response else text[:max_new_tokens]
    except Exception as e:
        logger.warning(f"⚠️ ChatGLM 摘要生成失败: {e}，使用 fallback")
        return text[:max_new_tokens]


def _build_question_prompt(text: str, page_url: str) -> str:
    """构建"根据内容生成代表性用户问题"的提示词（vLLM 和 ChatGLM 共用，
    从 `generate_question_ChatGLM` 抽出，供 `generate_question_vllm_async`
    复用同一套提示词构建逻辑，避免两处实现漂移）。
    """
    category = infer_chunk_category(page_url)

    # 分类对应的提示词
    category_hints = {
        "规则类": "平台是否允许、规则约束、违规处理",
        "操作类": "如何操作、是否可用、使用方法",
        "信息类": "平台背景、产品定位、策略设计",
        "泛用类": "用户实际可能会问的问题",
    }
    hint = category_hints.get(category, category_hints["泛用类"])

    return (
        f"你是一个电商平台知识问答构建助手，请根据以下内容生成一个有实际价值的用户问题。\n"
        f"要求：\n"
        f"- 问题应体现「{hint}」；\n"
        f"- 禁止复述原文，应提炼操作、判断或咨询点；\n"
        f"- 只输出一个简体中文问题句，不加说明。\n"
        f"📂 来源路径：{page_url}\n"
        f"📄 内容：\n{text}"
    )


def generate_question_ChatGLM(
    text: str,
    page_url: str,
    model,
    tokenizer,
    max_new_tokens: int = 64,
    fallback_question: str = "该内容可构造相关业务问题",
) -> str:
    """使用 ChatGLM 根据文档内容生成一个代表性用户问题。"""
    text = text.strip().replace("\x00", "")
    prompt = _build_question_prompt(text, page_url)

    try:
        response = _chatglm_generate(model, tokenizer, prompt, max_new_tokens, 0.7, 0.9)
        return response if response else fallback_question
    except Exception as e:
        logger.warning(f"⚠️ ChatGLM 问题生成失败: {e}，使用 fallback")
        return fallback_question


async def generate_question_vllm_async(
    text: str,
    page_url: str,
    model: str = "glm",
    max_new_tokens: int = 64,
    fallback_question: str = "该内容可构造相关业务问题",
) -> str:
    """异步调用 vLLM 根据文档内容生成一个代表性用户问题。

    对齐同步版 `generate_question_ChatGLM` 的 prompt 与语义，供
    `text_process.generate_block_documents_async` 在 `gen_question=True`
    时并发生成 question 字段（修复此前异步路径完全不生成 question 的
    一致性 BUG，见代码审查报告 M1）。
    """
    text = text.strip().replace("\x00", "")
    prompt = _build_question_prompt(text, page_url)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        async with sem:
            start = time.time()
            result = await call_vllm_with_retry_weighted(payload, timeout=VLLM_TIMEOUT)
            question = result["choices"][0]["message"]["content"].strip()
            logger.debug(f"✅ vLLM 异步问题生成成功 (耗时 {time.time() - start:.2f}s)")
            return question or fallback_question
    except Exception as e:
        logger.error(f"⚠️ vLLM 异步问题生成失败: {e}，使用 fallback")
        return fallback_question


# ======================== Query 重写 ========================

def _build_history_content(dialogue: list) -> str:
    """将多轮对话列表格式化为历史对话文本。"""
    lines = []
    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        lines.append(f"{role}：{content}")
    return "\n".join(lines) + "\n"


def _strip_thinking_chain(text: str) -> str:
    """去除思维链标记（</think> 之后的内容）。"""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _should_skip_rewrite(final_query: str, dialogue: list) -> bool:
    """判断是否应跳过 Query 重写（过滤词或对话过短）。"""
    if final_query in REWRITE_BANNED_PHRASES:
        logger.debug(f"🔍 命中过滤词 Query 重写跳过：{final_query}")
        return True
    if len(dialogue) < 2:
        logger.debug(f"🔍 对话历史过短，跳过 Query 重写：{final_query}")
        return True
    return False


def rewrite_query_ChatGLM(dialogue: list, final_query: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    """使用 ChatGLM 本地模型重写多轮对话中的模糊问题。"""
    fallback = final_query

    prompt = (
        "你是一个电商平台智能客服的对话清晰化助手。\n"
        "用户提出的问题可能存在复杂指代、上下文依赖或表达模糊等问题。\n"
        "你需要根据多轮历史对话，丰富润色用户的当前问题，"
        "使其成为一个清晰、完整的独立问题。\n\n"
        "下面是要求：\n"
        "- 准确解析用户真实意图，使得这个独立问题尽量完整，尽可能包含所有信息；\n"
        "- 问题越丰富越好，特别是要捕捉到关键的指代，场景，特别针对的问题和例子等等；\n"
        "- 不可以捏造不存在的信息；\n"
        "- 不添加解释、注释、引导语等，只输出润色后的问题句。\n\n"
        "下面是历史对话：\n"
    )
    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        prompt += f"{role}：{content}\n"
    prompt += f"用户当前问题是：{final_query.strip()}\n请你遵循要求润色为一个清晰、完整的独立问题："

    try:
        response = _chatglm_generate(model, tokenizer, prompt, max_new_tokens, 0.4, 0.8)
        return response if response else fallback
    except Exception as e:
        logger.warning(f"⚠️ ChatGLM 重写失败: {e}，返回原问题")
        return fallback


def rewrite_query_vllm(dialogue: list, final_query: str, model: str = "glm", max_new_tokens: int = 128) -> str:
    """同步调用 vLLM 重写多轮对话中的模糊问题。"""
    fallback = final_query
    prompt = (
        "你是一个问题重写API，只会重写优化或复读用户的问题。\n"
        "用户提出的问题可能存在复杂指代、上下文依赖或表达模糊等问题。\n"
        "你需要根据多轮历史对话，丰富润色用户的当前问题，"
        "使其成为一个清晰、完整的独立问题。\n\n"
        "下面是要求：\n"
        "- 准确解析用户真实意图，从历史中发掘指代和意图，"
        "使得这个独立问题尽量完整，尽可能包含所有信息关键词,"
        "特别是要补充关键的动机，指代和场景"
        "（润色后的问题至少涉及：用户面对什么前置情况，强调了什么限制，有什么疑问等），"
        "但不可以捏造不存在的信息；\n"
        "- 如果当前问题本身是 1.非问题（如寒暄、指令、灌水），"
        "2. 非技术性问题（身份询问、无关问题），不进行润色直接返回原句子；\n"
        "- 一定不可以回答用户问题，你只关注问题本身的润色；\n"
        "- 不添加解释、注释、引导语等，只输出润色后的问题句。\n\n"
        "下面是历史对话：\n"
    )
    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "系统"
        content = turn.get("text", "").replace("\n", " ").strip()
        prompt += f"{role}：{content}\n"
    prompt += f"用户当前问题是：{final_query.strip()}\n请你遵循要求润色为一个清晰、完整的独立问题："

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
        "top_p": 0.8,
    }
    try:
        start = time.time()
        api_url = CONFIG.get("vllm_api_url", "http://localhost:8011/v1/chat/completions")
        response = requests.post(api_url, json=payload, timeout=VLLM_TIMEOUT)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        logger.debug(f"✅ 重写成功（耗时 {time.time() - start:.2f}s）")
        return result if result else fallback
    except Exception as e:
        logger.warning(f"⚠️ vLLM 重写失败: {e}")
        return fallback


# ======================== 负载均衡 ========================

def weighted_sample_without_replacement(servers: List[Dict], tried: Set[str]) -> Optional[str]:
    """在未尝试服务器中按权重随机采样一个。"""
    candidates = [(s["url"], s.get("weight", 1)) for s in servers if s["url"] not in tried]
    if not candidates:
        return None
    urls, weights = zip(*candidates)
    return random.choices(urls, weights=weights, k=1)[0]


async def call_vllm_with_retry_weighted(payload: dict, timeout: int = 15, max_retries: Optional[int] = None) -> dict:
    """带权重的 vLLM 异步调用，失败自动重试。"""
    if aiohttp is None:  # [Optimized] aiohttp 未安装时提前给出明确错误
        raise RuntimeError("aiohttp 未安装，异步 LLM 调用不可用。请执行 pip install aiohttp")
    tried_urls = set()
    retries = max_retries or len(VLLM_SERVERS)
    errors = []

    for _ in range(retries):
        api_url = weighted_sample_without_replacement(VLLM_SERVERS, tried_urls)
        if api_url is None:
            break
        tried_urls.add(api_url)

        try:
            logger.debug(f"🚀 请求 vLLM: {api_url}")
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=timeout) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            logger.warning(f"⚠️ vLLM 请求失败: {api_url} → {e}")
            errors.append((api_url, str(e)))
            continue

    error_msg = "所有 vLLM 实例请求失败: " + "; ".join(f"{url}: {err}" for url, err in errors)
    raise RuntimeError(error_msg)


# ======================== 异步 Query 重写 ========================

async def rewrite_query_vllm_async(dialogue: list, final_query: str, model: str = "glm", max_new_tokens: int = 1024) -> str:
    """异步调用 vLLM 重写多轮对话中的模糊问题。

    Args:
        dialogue: 多轮对话历史，格式 [{"speaker": "user", "text": "..."}, ...]
        final_query: 当前用户的模糊问题
        model: vLLM 服务中注册的模型名
        max_new_tokens: 最大生成 token 数

    Returns:
        重写后的清晰问题；若无需重写则返回原问题
    """
    fallback = final_query

    if _should_skip_rewrite(final_query, dialogue):
        return fallback

    history = _build_history_content(dialogue)
    current = f"当前问题：{final_query.strip()}"

    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{history}\n{current}\n请你根据任务规则输出重写结果。"},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.5,
        "top_k": -1,
    }

    try:
        async with sem:
            start = time.time()
            result = await call_vllm_with_retry_weighted(payload, timeout=VLLM_TIMEOUT)
            rewritten = _strip_thinking_chain(result["choices"][0]["message"]["content"])
            logger.debug(f"{final_query} 重写成功，用时 {time.time() - start:.2f}s，结果：{rewritten}")
            return rewritten or fallback
    except Exception as e:
        logger.error(f"⚠️ 重写请求失败，返回原问题：{e}")
        return fallback


async def rewrite_query_ollama_async(dialogue: list, final_query: str, model: str = "qwen2.5:32b", max_new_tokens: int = 1024) -> str:
    """异步调用 Ollama 重写多轮对话中的模糊问题。"""
    fallback = final_query

    if _should_skip_rewrite(final_query, dialogue):
        return fallback

    history = _build_history_content(dialogue)
    current = f"当前问题：{final_query.strip()}"
    full_input = f"{history}\n{current}\n请你根据任务规则输出重写结果。"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": full_input},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.5,
    }

    try:
        if aiohttp is None:  # [Optimized] aiohttp 未安装时提前给出明确错误
            raise RuntimeError("aiohttp 未安装，异步 Query 重写不可用。请执行 pip install aiohttp")
        async with sem:
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_API_URL, json=payload, timeout=60) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    rewritten = _strip_thinking_chain(result["choices"][0]["message"]["content"])
                    logger.debug(f"{final_query} 重写成功，用时 {time.time() - start:.2f}s，结果：{rewritten}")
                    return rewritten or fallback
    except Exception as e:
        logger.error(f"⚠️ Ollama 请求失败，返回原问题：{e}")
        return fallback


# ======================== 向量嵌入 ========================

def get_embedding_from_vllm(text: str, url: str = None) -> list:
    """从 vLLM 服务获取文本嵌入向量。

    Args:
        text: 待嵌入的文本
        url: vLLM embedding 服务地址，若为 None 则从 CONFIG 读取

    Returns:
        嵌入向量（浮点数列表）
    """
    url = url or CONFIG.get("embed_api_url", "http://localhost:8010/v1/embeddings")
    payload = {
        "model": CONFIG.get("embed_model", "Qwen/Qwen3-Embedding-4B"),
        "input": [text],
    }
    resp = requests.post(url, json=payload, timeout=CONFIG.get("vllm_timeout", 60))
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


async def get_embeddings_from_vllm_async(texts: list, url: str, timeout: int = 10) -> list:
    """异步批量获取文本嵌入向量。"""
    from utils.config import get_aiohttp_session

    async def _fetch(text: str) -> list:
        payload = {"input": text}
        async with sem:
            session = await get_aiohttp_session()
            async with session.post(url, json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["data"][0]["embedding"]

    tasks = [asyncio.create_task(_fetch(txt)) for txt in texts]
    return await asyncio.gather(*tasks)
