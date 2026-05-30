from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig

# 原生核心解耦层绝对路径规范化引入
from astrbot.core.agent.message import Message
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
import json
import asyncio


@register(
    "astrbot_plugin_context_compressor", "kitakita0421", "无感压缩上下文", "2.1.2"
)
class CanonicalCompressorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.token_counter = EstimateTokenCounter()
        logger.info("[无感压缩上下文] 2.1.2 生产交付版初始化成功。")

    @filter.command("compress")
    async def compress_command(self, event: AstrMessageEvent):
        """用户手动立刻对当前会话执行上下文压缩总结"""
        logger.info(
            f"[无感压缩上下文] 侦听到用户 {event.get_sender_id()} 手动输入 /compress 指令"
        )
        yield event.plain_result("正在分析物理路由矩阵并执行同步压缩...")
        if await self._run_canonical_compression(event, force_trigger=True):
            yield event.plain_result("✨ 专属策略同步压缩成功！历史结构已原地更新。")
        else:
            yield event.plain_result(
                "⚠️ 压缩未触发。当前非系统消息原文轮数过少，未达到最低保留额度。"
            )

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message(self, event: AstrMessageEvent, *args, **kwargs):
        """核心管道消息拦截点，直接分派协程进行后台异步状态机轮询"""
        asyncio.create_task(self._background_wait_and_compress(event))

    async def _background_wait_and_compress(self, event: AstrMessageEvent):
        """后台独立状态机：秒级轮询侦听 AI 回复是否完整落库写入 SQLite"""
        conv_mgr = self.context.conversation_manager
        uid = event.unified_msg_origin
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)

        initial_count = 0
        conversation = await conv_mgr.get_conversation(uid, curr_cid)
        if conversation and conversation.history:
            try:
                initial_count = len(json.loads(conversation.history))
            except Exception:
                pass

        for i in range(25):
            await asyncio.sleep(1.0)
            conversation = await conv_mgr.get_conversation(uid, curr_cid)
            if not conversation or not conversation.history:
                continue

            try:
                current_history = json.loads(conversation.history)
                current_count = len(current_history)

                if (
                    current_count > initial_count
                    and current_history[-1].get("role") == "assistant"
                ):
                    logger.info(
                        f"[无感压缩上下文] 🔔 [后台任务] AI回复已安全落库 (消息数: {initial_count} -> {current_count})。拉起自适应轮数检测..."
                    )
                    break
            except Exception:
                pass

            if i == 24:
                logger.warning(
                    "[无感压缩上下文] ⏳ [后台任务] 轮询达到安全时限，强行进入检测管道..."
                )

        try:
            await self._run_canonical_compression(event, force_trigger=False)
        except Exception as run_err:
            logger.error(
                f"[无感压缩上下文] ❌ [后台任务] 执行核心自适应压缩时崩溃: {run_err}",
                exc_info=True,
            )

    async def _run_canonical_compression(
        self, event: AstrMessageEvent, force_trigger: bool = False
    ) -> bool:
        conv_mgr = self.context.conversation_manager
        uid = event.unified_msg_origin
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)
        conversation = await conv_mgr.get_conversation(uid, curr_cid)

        if not conversation or not conversation.history:
            return False

        try:
            raw_history = json.loads(conversation.history)
            msg_obj_list = [
                Message(
                    role=m.get("role"),
                    content=m.get("content"),
                    tool_calls=m.get("tool_calls"),
                )
                for m in raw_history
            ]
        except Exception:
            return False

        # 1. 物理 3 维特征确定性规范化提取
        runtime_provider = None
        if hasattr(event, "raw_event") and isinstance(event.raw_event, dict):
            runtime_provider = event.raw_event.get(
                "provider_id"
            ) or event.raw_event.get("chat_provider_id")

        if not runtime_provider:
            runtime_provider = await self.context.get_current_chat_provider_id(umo=uid)

        runtime_provider = str(runtime_provider).strip()
        runtime_platform_id = event.get_platform_id()
        runtime_group_id = (
            str(event.message_obj.group_id).strip()
            if (event.message_obj and event.message_obj.group_id)
            else ""
        )
        runtime_chat_type = "group" if runtime_group_id else "private"

        logger.info(
            f"[无感压缩上下文] 📊 环境核对矩阵: Model='{runtime_provider}', BotInstance='{runtime_platform_id}', Type='{runtime_chat_type}'"
        )

        try:
            max_turns = int(self.config.get("max_conversation_length", 40))
            max_tokens = int(self.config.get("max_context_tokens", 16000))
            keep_recent = int(self.config.get("keep_recent", 2))
        except (ValueError, TypeError):
            max_turns, max_tokens, keep_recent = 40, 16000, 2

        # 2. 物理特异性路由打分器
        special_rules = self.config.get("special_rules", [])
        best_rule, best_specificity_score = None, -1
        for rule in special_rules:
            if (
                rule.get("target_provider", "").strip()
                and rule.get("target_provider", "").strip() != runtime_provider
            ):
                continue
            if (
                rule.get("target_platform_id", "").strip()
                and rule.get("target_platform_id", "").strip() != runtime_platform_id
            ):
                continue
            if (
                rule.get("chat_type", "").strip()
                and rule.get("chat_type", "").strip() != runtime_chat_type
            ):
                continue

            current_score = sum(
                [
                    bool(rule.get("target_provider")),
                    bool(rule.get("target_platform_id")),
                    bool(rule.get("chat_type")),
                ]
            )
            if current_score > best_specificity_score:
                best_specificity_score = current_score
                best_rule = rule

        if best_rule and best_specificity_score > 0:
            try:
                max_turns = int(best_rule.get("max_turns", max_turns))
                max_tokens = int(best_rule.get("max_tokens", max_tokens))
                keep_recent = int(best_rule.get("keep_recent", keep_recent))
            except (ValueError, TypeError):
                pass

        keep_recent_msg_count = keep_recent * 2

        # 3. 剥离头部 System
        first_non_system = 0
        for i, msg in enumerate(msg_obj_list):
            if msg.role != "system":
                first_non_system = i
                break

        non_system_objs = msg_obj_list[first_non_system:]
        current_turns = len(non_system_objs) // 2
        logger.info(
            f"[无感压缩上下文] 轮数审计: 当前物理对话轮数 = {current_turns} 轮 (阈值限制 = {max_turns} 轮)"
        )

        if len(non_system_objs) <= keep_recent_msg_count or len(non_system_objs) <= 0:
            return False

        # 阈值校验
        should_compress = force_trigger
        if not should_compress:
            if current_turns >= max_turns:
                should_compress = True
                logger.info(
                    f"[无感压缩上下文] 🚦 满足水线: 物理轮数达到上限 ({current_turns} >= {max_turns} 轮)"
                )
            elif max_tokens > 0:
                estimated_tokens = self.token_counter.count_tokens(non_system_objs)
                if estimated_tokens >= max_tokens:
                    should_compress = True
                    logger.info(
                        f"[无感压缩上下文] 🚦 满足水线: 估算 Token 突破上限 ({estimated_tokens} >= {max_tokens})"
                    )

        if not should_compress:
            return False

        # 4. 回溯安全 User 切分点索引
        split_index = len(non_system_objs) - keep_recent_msg_count
        if split_index >= len(non_system_objs):
            split_index = len(non_system_objs) - 1

        while split_index > 0 and non_system_objs[split_index].role != "user":
            split_index -= 1

        if split_index <= 0:
            return False

        # 5. 多级总结模型路由
        final_summary_provider = (
            best_rule.get("rule_summary_provider", "").strip()
            if (best_rule and best_specificity_score > 0)
            else None
        )
        if not final_summary_provider:
            final_summary_provider = runtime_provider

        # 6. 生成总结
        messages_to_summarize = non_system_objs[:split_index]
        summary_instruction = self.config.get("summary_instruction", "").strip() or (
            "Based on our full conversation history, produce a concise summary of key takeaways and/or project progress.\n"
            "1. Systematically cover all core topics discussed and the final conclusion/outcome for each; clearly highlight the latest primary focus.\n"
            "2. If any tools were used, summarize tool usage (total call count) and extract the most valuable insights from tool outputs.\n"
            "3. If there was an initial user goal, state it first and describe the current progress/status.\n"
            "4. Write the summary in the user's language.\n"
        )

        llm_payload = messages_to_summarize + [
            Message(role="user", content=summary_instruction)
        ]

        logger.info(
            f"[无感压缩上下文] 🚀 后台指派总结模型: '{final_summary_provider}' 归档前 {split_index // 2} 轮历史..."
        )
        llm_resp = await self.context.llm_generate(
            chat_provider_id=final_summary_provider, prompt=None, contexts=llm_payload
        )
        if not llm_resp or not llm_resp.completion_text:
            logger.error(
                f"[无感压缩上下文] 调取模型 {final_summary_provider} 生成总结失败。"
            )
            return False

        summary_content = llm_resp.completion_text

        # 7. 持久层原生无损切片回写 SQLite
        system_dicts = raw_history[:first_non_system]
        non_system_dicts = raw_history[first_non_system:]
        recent_dicts = non_system_dicts[split_index:]

        new_history = (
            system_dicts
            + [
                {
                    "role": "user",
                    "content": f"Our previous history conversation summary: {summary_content}",
                },
                {
                    "role": "assistant",
                    "content": "Acknowledged the summary of our previous conversation history.",
                },
            ]
            + recent_dicts
        )

        await conv_mgr.update_conversation(
            unified_msg_origin=uid, conversation_id=curr_cid, history=new_history
        )
        logger.info(
            f"[无感压缩上下文] 🎉 🎉 🎉 成功！已通过模型 [{final_summary_provider}] 原地重构会话历史。"
        )
        return True
