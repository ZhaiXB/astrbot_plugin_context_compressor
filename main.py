from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger, AstrBotConfig

# 完美引入官方核心强类型骨架与序列化/反序列化无损算子
from astrbot.core.agent.message import (
    Message, 
    bind_checkpoint_messages, 
    dump_messages_with_checkpoints
)
from astrbot.core.agent.context.token_counter import EstimateTokenCounter
import json
import asyncio
from typing import Any

@register("astrbot_plugin_context_compressor", "kitakita0421", "无感压缩上下文", "2.2.4")
class CanonicalCompressorPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.token_counter = EstimateTokenCounter()
        
        # 运行时高阶立体维度的镜像快照缓存
        self.system_prompt_snapshot = {}  # CID -> 完美系统提示词
        self.tools_snapshot = {}          # CID -> 完美 ToolSet 工具集
        self.kwargs_snapshot = {}         # CID -> 全量运行时模型控制超参数
        
        # 🌟 核心防患未然：异步并发重叠防重锁（确保同一会话同一时间只有一个总结任务在飞）
        self.compressing_cids = set()     # 记录正在后台执行执行总结压缩的 CID 集合
        
        logger.info("[无感压缩上下文] v2.2.4 终极闭环无损强类型对象流版载入成功。")

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

                # 动态全量超参数搜刮（不改变、不碰消息体，只剥离数值/控制参数）
                extracted_kwargs = {}
                standard_fields = {"prompt", "contexts", "system_prompt", "func_tool", "image_urls", "audio_urls"}
                for k, v in getattr(req, "__dict__", {}).items():
                    if k not in standard_fields and not k.startswith("_") and v is not None:
                        extracted_kwargs[k] = v
                if hasattr(req, "model_dump"):
                    try:
                        for k, v in req.model_dump().items():
                            if k not in standard_fields and not k.startswith("_") and v is not None:
                                extracted_kwargs[k] = v
                    except Exception: pass
                self.kwargs_snapshot[curr_cid] = extracted_kwargs
        except Exception:
            pass

    @filter.command("compress")
    async def compress_command(self, event: AstrMessageEvent):
        """手动触发清洗指令"""
        yield event.plain_result("正在同步压缩历史上下文对象...")
        if await self._run_canonical_compression(event, force_trigger=True):
            yield event.plain_result("✨ 原生对象流压缩重构成功！检查点与并发增量已无损合流。")
        else:
            yield event.plain_result("⚠️ 压缩未触发。当前会话正处于后台压缩中，或未达保留水线。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message(self, event: AstrMessageEvent, *args, **kwargs):
        """分派后台异步状态机轮询"""
        asyncio.create_task(self._background_wait_and_compress(event))

    async def _background_wait_and_compress(self, event: AstrMessageEvent):
        """秒级轮询落库状态机"""
        conv_mgr = self.context.conversation_manager
        uid = event.unified_msg_origin
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)
        
        initial_count = 0
        conversation = await conv_mgr.get_conversation(uid, curr_cid)
        if conversation and conversation.history:
            try: initial_count = len(json.loads(conversation.history))
            except Exception: pass

        for i in range(25):
            await asyncio.sleep(1.0)
            conversation = await conv_mgr.get_conversation(uid, curr_cid)
            if not conversation or not conversation.history: continue
                
            try:
                current_history = json.loads(conversation.history)
                current_count = len(current_history)
                
                if current_count > initial_count and current_history[-1].get("role") == "assistant":
                    break
            except Exception:
                pass

        try:
            await self._run_canonical_compression(event, force_trigger=False)
        except Exception as run_err:
            logger.error(f"[无感压缩上下文] 后台自适应压缩发生异常: {run_err}", exc_info=True)

    def _resolve_runtime_context(self, event: AstrMessageEvent, conversation) -> tuple[str, str]:
        uid = event.unified_msg_origin
        provider, persona = "", ""
        for obj in [event, getattr(event, "raw_event", None), event.message_obj, conversation]:
            if not obj: continue
            if isinstance(obj, dict):
                provider = provider or obj.get("provider_id") or obj.get("chat_provider_id")
                persona = persona or obj.get("persona_id") or obj.get("profile_id")
            else:
                provider = provider or getattr(obj, "provider_id", None) or getattr(obj, "chat_provider_id", None)
                persona = persona or getattr(obj, "persona_id", None) or getattr(obj, "profile_id", None)
        return str(provider or "deepseek/deepseek-v4-flash").strip(), str(persona).strip()

    async def _run_canonical_compression(self, event: AstrMessageEvent, force_trigger: bool = False) -> bool:
        conv_mgr = self.context.conversation_manager
        uid = event.unified_msg_origin
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)
        
        # 🌟 核心防患未然：如果当前会话已经在执行后台压缩，后续的重叠重试协程直接就地安全退避
        if curr_cid in self.compressing_cids:
            logger.debug(f"[无感压缩上下文] 会话 {curr_cid} 正在后台压缩中，跳过本次重叠触发。")
            return False

        conversation = await conv_mgr.get_conversation(uid, curr_cid)
        if not conversation or not conversation.history:
            return False

        try:
            raw_history = json.loads(conversation.history)
        except Exception as e:
            logger.error(f"[无感压缩上下文] 解析会话历史记录 JSON 失败: {e}")
            return False

        # 1. 利用官方强类型装配算子将生历史一键转换为 Message 对象数组
        all_messages = bind_checkpoint_messages(raw_history)
        dialogue_messages = [m for m in all_messages if m.role != "system"]

        # 提取当前特征
        runtime_provider, _ = self._resolve_runtime_context(event, conversation)
        if runtime_provider == "deepseek/deepseek-v4-flash":
            runtime_provider = await self.context.get_current_chat_provider_id(umo=uid)
        runtime_provider = str(runtime_provider).strip()
        runtime_platform_id = event.get_platform_id()
        runtime_group_id = str(event.message_obj.group_id).strip() if (event.message_obj and event.message_obj.group_id) else ""
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
            if rule.get("target_provider", "").strip() and rule.get("target_provider", "").strip() != runtime_provider: continue
            if rule.get("target_platform_id", "").strip() and rule.get("target_platform_id", "").strip() != runtime_platform_id: continue
            if rule.get("chat_type", "").strip() and rule.get("chat_type", "").strip() != runtime_chat_type: continue
            current_score = sum([bool(rule.get("target_provider")), bool(rule.get("target_platform_id")), bool(rule.get("chat_type"))])
            if current_score > best_specificity_score:
                best_specificity_score = current_score; best_rule = rule

        if best_rule and best_specificity_score > 0:
            try:
                max_turns = int(best_rule.get("max_turns", max_turns))
                max_tokens = int(best_rule.get("max_tokens", max_tokens))
                keep_recent = int(best_rule.get("keep_recent", keep_recent))
            except (ValueError, TypeError): pass

        # 2. 精准审计轮数
        user_indices = [i for i, m in enumerate(dialogue_messages) if m.role == "user"]
        current_turns = len(user_indices)

        if len(dialogue_messages) <= keep_recent * 2 or current_turns < max_turns:
            return False

        # 阈值判定
        should_compress = force_trigger
        if not should_compress:
            if current_turns >= max_turns:
                should_compress = True
            elif max_tokens > 0 and self.token_counter.count_tokens(dialogue_messages) >= max_tokens:
                should_compress = True

        if not should_compress:
            return False

        # 🌟 成功突破阈值关卡，正式对当前 CID 上内存会话锁
        self.compressing_cids.add(curr_cid)

        try:
            # 3. 完美的物理对象切分
            split_user_msg = dialogue_messages[user_indices[-keep_recent]]
            split_index_in_all = all_messages.index(split_user_msg)

            to_summarize_messages = all_messages[:split_index_in_all]
            recent_messages = all_messages[split_index_in_all:]

            # 剥离旧 system 消息
            to_summarize_messages = [m for m in to_summarize_messages if m.role != "system"]

            # 确定路由模型
            final_summary_provider = best_rule.get("rule_summary_provider", "").strip() if (best_rule and best_specificity_score > 0) else None
            if not final_summary_provider: final_summary_provider = runtime_provider

            # 4. 完美继承快照特征
            base_system_prompt = self.system_prompt_snapshot.get(curr_cid)
            base_tools = self.tools_snapshot.get(curr_cid)
            base_kwargs = self.kwargs_snapshot.get(curr_cid) or {}
            
            if not base_system_prompt:
                base_system_prompt = "You are a helpful and precise AI partner."

            # 5. 原汁原味对象投递
            summary_instruction = self.config.get("summary_instruction", "").strip() or (
                "Based on our full conversation history, produce a concise summary of key takeaways.\n"
                "Write the summary in the user's language."
            )
            
            payload_contexts = to_summarize_messages + [
                Message(role="user", content=f"{summary_instruction}\n\nPlease output the summary directly.")
            ]

            llm_resp = await self.context.llm_generate(
                chat_provider_id=final_summary_provider, 
                system_prompt=base_system_prompt,  
                tools=base_tools,                  
                contexts=payload_contexts,
                **base_kwargs  
            )
            
            if not llm_resp or not llm_resp.completion_text:
                return False

            summary_content = llm_resp.completion_text

            # 🌟 核心增量补丁算子 (Delta Patching)：重新打捞数据库
            refreshed_conversation = await conv_mgr.get_conversation(uid, curr_cid)
            if refreshed_conversation and refreshed_conversation.history:
                try:
                    latest_raw_history = json.loads(refreshed_conversation.history)
                    latest_messages = bind_checkpoint_messages(latest_raw_history)
                    
                    if len(latest_messages) > len(all_messages):
                        delta_messages = latest_messages[len(all_messages):]
                        recent_messages = recent_messages + delta_messages
                        logger.info(f"[无感压缩上下文] ⚡ 检测到总结的 3 秒内发生高频手快新消息！已就地追加 {len(delta_messages)} 条增量补丁，无损合流。")
                except Exception as ex:
                    logger.warning(f"[无感压缩上下文] 提取增量历史补丁时退避: {ex}")

            # 6. 原生对象级落库组装
            system_messages = [m for m in all_messages if m.role == "system"]
            summary_messages = [
                Message(role="user", content=f"Our previous history conversation summary: {summary_content}"),
                Message(role="assistant", content="Acknowledged the summary of our previous conversation history.")
            ]
            
            new_messages = system_messages + summary_messages + recent_messages
            new_history_dicts = dump_messages_with_checkpoints(new_messages)

            # 针对 SQLite 物理锁冲突的指数避让退避机制
            for attempt in range(5):
                try:
                    await conv_mgr.update_conversation(unified_msg_origin=uid, conversation_id=curr_cid, history=new_history_dicts)
                    break
                except Exception as db_err:
                    if "locked" in str(db_err).lower() and attempt < 4:
                        sleep_time = 0.2 * (2 ** attempt)
                        logger.warning(f"[无感压缩上下文] 遇到数据库锁冲突，正在执行第 {attempt+1} 次指数退避 ({sleep_time}s)...")
                        await asyncio.sleep(sleep_time)
                    else:
                        raise db_err

            return True
            
        finally:
            # 🌟 无论总结成功还是最终报错退出，强制在析构阶段彻底清除内存锁，让下一次压缩管道畅通无阻
            self.compressing_cids.discard(curr_cid)
