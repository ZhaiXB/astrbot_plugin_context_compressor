from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig

# 完美引入官方核心强类型骨架 with 序列化/反序列化无损算子 (带防断裂降级保护)
try:
    from astrbot.core.agent.message import (
        Message,
        bind_checkpoint_messages,
        dump_messages_with_checkpoints,
    )
    HAS_OFFICIAL_CHECKPOINT_OPERATORS = True
except ImportError:
    HAS_OFFICIAL_CHECKPOINT_OPERATORS = False
    logger.warning("[无感压缩] ⚠️ 无法导入官方 Checkpoint 算子，已自动降级为纯字典兼容层！")

    class Message:
        def __init__(self, role: str, content: Any = None, **kwargs):
            self.role = role
            self.content = content
            self._checkpoint_after = None
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def model_dump(self):
            res = {"role": self.role, "content": self.content}
            for k, v in self.__dict__.items():
                if not k.startswith("_") and k not in res:
                    res[k] = v
            return res

    def bind_checkpoint_messages(history: list[dict]) -> list[Message]:
        messages = []
        for item in history:
            if item.get("role") == "_checkpoint":
                if messages:
                    messages[-1]._checkpoint_after = item.get("content")
                continue
            messages.append(Message(**item))
        return messages

    def dump_messages_with_checkpoints(messages: list[Message]) -> list[dict]:
        dumped = []
        for m in messages:
            dumped.append(m.model_dump())
            if getattr(m, "_checkpoint_after", None) is not None:
                dumped.append({
                    "role": "_checkpoint",
                    "content": m._checkpoint_after
                })
        return dumped

try:
    from astrbot.core.agent.context.token_counter import EstimateTokenCounter
except ImportError:
    logger.warning("[无感压缩] ⚠️ 无法导入官方 EstimatorTokenCounter 算子，已自动降级为字数估算兼容层！")
    class EstimateTokenCounter:
        def count_tokens(self, messages: list) -> int:
            total = 0
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            total += len(part.get("text", ""))
                        elif hasattr(part, "text"):
                            total += len(part.text)
            return total // 2

import json
import asyncio
from typing import Any


@register(
    "astrbot_plugin_context_compressor", "kitakita0421", "无感压缩上下文", "2.2.6"
)
class CanonicalCompressorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.token_counter = EstimateTokenCounter()

        # 完全抽象设计：使用中立的物理占位符替代特定模型硬编码名称
        self.default_provider_placeholder = "default/placeholder"

        # 运行时高阶立体维度的镜像快照缓存
        self.system_prompt_snapshot = {}  # CID -> 完美系统提示词
        self.tools_snapshot = {}  # CID -> 完美 ToolSet 工具集
        self.kwargs_snapshot = {}  # CID -> 全量运行时模型控制超参数

        # 核心防患未然：异步并发重叠防重锁（确保同一会话同一时间只有一个总结任务在飞）
        self.compressing_cids = set()  # 记录正在后台执行总结压缩的 CID 集合
        self.polling_cids = set()  # 记录正在进行后台等待的 CID 集合，彻底消灭冗余轮询协程

        logger.info("[无感压缩] v2.2.6 极致通用对象流版载入成功。")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: Any):
        """主管道拦截器：静默截获并无损提取完全态的 System Prompt、Tools 以及未知全量超参数"""
        try:
            conv_mgr = self.context.conversation_manager
            uid = event.unified_msg_origin
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
            if curr_cid:
                if getattr(req, "system_prompt", None):
                    self.system_prompt_snapshot[curr_cid] = req.system_prompt
                if getattr(req, "func_tool", None):
                    self.tools_snapshot[curr_cid] = req.func_tool

                # 动态全量超参数搜刮（只搜刮控制型超参数，必须把所有非JSON序列化对象及模型名彻底黑名单过滤）
                extracted_kwargs = {}
                standard_fields = {
                    "prompt",
                    "contexts",
                    "system_prompt",
                    "func_tool",
                    "image_urls",
                    "audio_urls",
                    "session_id",
                    "conversation",  # 必须排除数据库实体模型，避免触发 sql 序列化崩溃
                    "tool_calls_result",  # 必须排除工具调用复杂模型
                    "extra_user_content_parts",  # 必须排除多模态缓存复杂类
                    "model",  # 必须排除 model 属性，保证压缩任务路由的独立性不受原始请求偏置干扰
                }
                for k, v in getattr(req, "__dict__", {}).items():
                    if (
                        k not in standard_fields
                        and not k.startswith("_")
                        and v is not None
                    ):
                        extracted_kwargs[k] = v
                if hasattr(req, "model_dump"):
                    try:
                        for k, v in req.model_dump().items():
                            if (
                                k not in standard_fields
                                and not k.startswith("_")
                                and v is not None
                            ):
                                extracted_kwargs[k] = v
                    except Exception:
                        pass
                self.kwargs_snapshot[curr_cid] = extracted_kwargs
        except Exception:
            pass

    @filter.command("compress")
    async def compress_command(self, event: AstrMessageEvent):
        """手动触发清洗指令"""
        yield event.plain_result("正在同步压缩历史上下文对象...")
        if await self._run_canonical_compression(event, force_trigger=True):
            yield event.plain_result(
                "✨ 原生对象流压缩重构成功！检查点与并发增量已无损合流。"
            )
        else:
            yield event.plain_result(
                "⚠️ 压缩未触发。当前会话正处于后台压缩中，或未达保留水线。"
            )

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message(self, event: AstrMessageEvent, *args, **kwargs):
        """分派后台异步状态机轮询"""
        conv_mgr = self.context.conversation_manager
        uid = event.unified_msg_origin
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)
        if not curr_cid:
            return

        # 引入防重复轮询锁机制，每个 CID 在同一时刻有且仅有一个轮询器在跑，大幅节约空转开销
        if curr_cid in self.polling_cids:
            return

        self.polling_cids.add(curr_cid)
        asyncio.create_task(self._background_wait_and_compress(event, curr_cid))

    async def _background_wait_and_compress(self, event: AstrMessageEvent, curr_cid: str):
        """秒级轮询落库状态机"""
        try:
            conv_mgr = self.context.conversation_manager
            uid = event.unified_msg_origin

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
                        break
                except Exception:
                    pass

            try:
                await self._run_canonical_compression(event, force_trigger=False)
            except Exception as run_err:
                logger.error(
                    f"[无感压缩] 后台自适应压缩发生异常: {run_err}", exc_info=True
                )
        finally:
            # 无论等待完成还是抛出异常，在析构时无条件释放会话的 Polling 锁
            self.polling_cids.discard(curr_cid)

    def _resolve_runtime_context(
        self, event: AstrMessageEvent, conversation
    ) -> tuple[str, str]:
        """环境特征纯净化规范提取器"""
        # ✅ 已修正：完美移除未使用的 uid 变量局部赋值，彻底解决 Ruff F841 警告
        provider, persona = "", ""
        for obj in [
            event,
            getattr(event, "raw_event", None),
            event.message_obj,
            conversation,
        ]:
            if not obj:
                continue
            if isinstance(obj, dict):
                provider = (
                    provider or obj.get("provider_id") or obj.get("chat_provider_id")
                )
                persona = persona or obj.get("persona_id") or obj.get("profile_id")
            else:
                provider = (
                    provider
                    or getattr(obj, "provider_id", None)
                    or getattr(obj, "chat_provider_id", None)
                )
                persona = (
                    persona
                    or getattr(obj, "persona_id", None)
                    or getattr(obj, "profile_id", None)
                )

        return str(provider or self.default_provider_placeholder).strip(), str(
            persona
        ).strip()

    async def _run_canonical_compression(
        self, event: AstrMessageEvent, force_trigger: bool = False
    ) -> bool:
        conv_mgr = self.context.conversation_manager
        uid = event.unified_msg_origin
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)

        # 核心防患未然：如果当前会话已经在执行后台压缩，后续的重叠重试协程直接就地安全退避
        if curr_cid in self.compressing_cids:
            logger.debug(
                f"[无感压缩] 会话 {curr_cid[-8:] if curr_cid else 'N/A'} 正在后台压缩中，跳过本次重叠触发。"
            )
            return False

        # 立即上锁，彻底消除 TOCTOU 并发竞态隐患
        self.compressing_cids.add(curr_cid)

        try:
            conversation = await conv_mgr.get_conversation(uid, curr_cid)
            if not conversation or not conversation.history:
                return False

            try:
                raw_history = json.loads(conversation.history)
            except Exception as e:
                logger.error(f"[无感压缩] 解析会话历史记录 JSON 失败: {e}")
                return False

            # 1. 照搬官方：利用强类型装配算子将生历史一键转换为 Message 对象数组
            all_messages = bind_checkpoint_messages(raw_history)
            dialogue_messages = [m for m in all_messages if m.role != "system"]

            # 提取当前特征
            runtime_provider, _ = self._resolve_runtime_context(event, conversation)

            # 格式化解耦判定：只要触发了内部抽象占位符，或者取出的模型格式不合法（缺少提供商前缀 /），一律强行向下刷新
            if (
                runtime_provider == self.default_provider_placeholder
                or "/" not in runtime_provider
            ):
                runtime_provider = await self.context.get_current_chat_provider_id(umo=uid)

            runtime_provider = str(runtime_provider).strip()
            runtime_platform_id = event.get_platform_id()
            runtime_group_id = (
                str(event.message_obj.group_id).strip()
                if (event.message_obj and event.message_obj.group_id)
                else ""
            )
            runtime_chat_type = "group" if runtime_group_id else "private"

            # 载入配置
            try:
                max_turns = int(self.config.get("max_conversation_length", 40))
                max_tokens = int(self.config.get("max_context_tokens", 16000))
                keep_recent = int(self.config.get("keep_recent", 2))
            except (ValueError, TypeError):
                max_turns, max_tokens, keep_recent = 40, 16000, 2

            # 路由网格规则碰撞
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

            # 极端边界防御：清洗 keep_recent 配置，保证其至少为 1
            keep_recent = max(1, keep_recent)

            # 2. 精准审计与 Token 统计：通过 Message 对象统计用户轮数与估算 Token
            user_indices = [i for i, m in enumerate(dialogue_messages) if m.role == "user"]
            current_turns = len(user_indices)
            current_tokens = self.token_counter.count_tokens(dialogue_messages)

            logger.info(
                f"[无感压缩] 会话 {curr_cid[-8:] if curr_cid else 'N/A'}: "
                f"当前 {current_turns} 轮/{current_tokens}T (阈值: {max_turns}轮/{max_tokens}T)"
            )

            # 极端边界防御：如果我们保留的最近轮数 >= 总用户轮数，说明没有老对话可压缩，直接就地返回 False
            if len(user_indices) <= keep_recent:
                return False

            if len(dialogue_messages) <= keep_recent * 2 or current_turns < max_turns:
                return False

            # 阈值判定
            should_compress = force_trigger
            if not should_compress:
                if current_turns >= max_turns:
                    should_compress = True
                elif (
                    max_tokens > 0
                    and current_tokens >= max_tokens
                ):
                    should_compress = True

            if not should_compress:
                return False

            # 3. 完美的物理对象切分 (因上面做了 user_indices 长度强校验，此处的 slice 边界绝对安全，不会 IndexError)
            split_user_msg = dialogue_messages[user_indices[-keep_recent]]
            split_index_in_all = all_messages.index(split_user_msg)

            to_summarize_messages = all_messages[:split_index_in_all]
            recent_messages = all_messages[split_index_in_all:]

            # 剥离旧 system 消息
            to_summarize_messages = [
                m for m in to_summarize_messages if m.role != "system"
            ]

            # 确定路由模型
            final_summary_provider = (
                best_rule.get("rule_summary_provider", "").strip()
                if (best_rule and best_specificity_score > 0)
                else None
            )
            if not final_summary_provider:
                final_summary_provider = runtime_provider

            # 4. 完美镜像继承快照特征
            base_system_prompt = self.system_prompt_snapshot.get(curr_cid)
            base_tools = self.tools_snapshot.get(curr_cid)
            base_kwargs = self.kwargs_snapshot.get(curr_cid) or {}

            if not base_system_prompt:
                logger.warning(f"[无感压缩] 会话 {curr_cid[-8:] if curr_cid else 'N/A'} 未截获 System Prompt 快照，使用兜底提示词(缓存命中率可能受影响)。")
                base_system_prompt = "You are a helpful and precise AI partner."

            # 5. 原汁原味对象投递
            summary_instruction = self.config.get(
                "summary_instruction", ""
            ).strip() or (
                "Based on our full conversation history, produce a concise summary of key takeaways and/or project progress.\n"
                "The primary goal of this summary is to enable seamless continuation of the work that follows.\n"
                "1. Systematically cover all core topics discussed and the final conclusion/outcome for each; clearly highlight the latest primary focus.\n"
                "2. If any tools were used, summarize tool usage (total call count) and extract the most valuable insights from tool outputs.\n"
                "3. If any materials (files, documents, code, references) were read during the conversation that may be helpful for subsequent work, list each one with its scope and path.\n"
                "4. If there was an initial user goal, state it first and describe the current progress/status.\n"
                "5. Write the summary in the user's language."
            )

            payload_contexts = to_summarize_messages + [
                Message(
                    role="user",
                    content=f"{summary_instruction}\n\nPlease output the summary directly.",
                )
            ]

            llm_resp = await self.context.llm_generate(
                chat_provider_id=final_summary_provider,
                system_prompt=base_system_prompt,
                tools=base_tools,
                contexts=payload_contexts,
                **base_kwargs,
            )

            if not llm_resp or not llm_resp.completion_text:
                return False

            summary_content = llm_resp.completion_text

            # 核心增量补丁算子 (Delta Patching)：重新打捞数据库
            refreshed_conversation = await conv_mgr.get_conversation(uid, curr_cid)
            if refreshed_conversation and refreshed_conversation.history:
                try:
                    latest_raw_history = json.loads(refreshed_conversation.history)
                    latest_messages = bind_checkpoint_messages(latest_raw_history)

                    if len(latest_messages) > len(all_messages):
                        delta_messages = latest_messages[len(all_messages) :]
                        recent_messages = recent_messages + delta_messages
                        logger.info(
                            f"[无感压缩] ⚡ 检测到 3 秒内发生高频新消息！已追加 {len(delta_messages)} 条增量补丁，无损合流。"
                        )
                except Exception as ex:
                    logger.warning(f"[无感压缩] 提取增量补丁退避: {ex}")

            # 6. 原生对象级落库组装
            system_messages = [m for m in all_messages if m.role == "system"]
            summary_messages = [
                Message(
                    role="user",
                    content=f"Our previous history conversation summary: {summary_content}",
                ),
                Message(
                    role="assistant",
                    content="Acknowledged the summary of our previous conversation history.",
                ),
            ]

            new_messages = system_messages + summary_messages + recent_messages
            new_history_dicts = dump_messages_with_checkpoints(new_messages)

            # 针对 SQLite 物理锁冲突的指数避让退避机制
            for attempt in range(5):
                try:
                    await conv_mgr.update_conversation(
                        unified_msg_origin=uid,
                        conversation_id=curr_cid,
                        history=new_history_dicts,
                    )
                    break
                except Exception as db_err:
                    if "locked" in str(db_err).lower() and attempt < 4:
                        sleep_time = 0.2 * (2**attempt)
                        logger.warning(f"[无感压缩] 数据库锁冲突，执行第 {attempt + 1} 次指数退避 ({sleep_time}s)...")
                        await asyncio.sleep(sleep_time)
                    else:
                        raise db_err

            return True

        finally:
            # 无论总结成功还是最终报错退出，强制在析构阶段彻底清除内存锁，让下一次压缩管道畅通无阻
            self.compressing_cids.discard(curr_cid)
