## processing

stage 0 for ready,    successful
stage 1 for ready,    successful
stage 2 for ready,    successful

waiting all worker send Handshake Received, successful

Logic for handling diffusion sleep and wakeup memory changes,  successful

Logic for handling llm and generation sleep and wakeup memory changes, successful

test pytest 5, successful

add AMD logic, successful


## next step

1. test AMD, pending












## NVIDIA 

RTX A5000 TP=2


### TEST 5

![alt text](vram_recycling_unittest_5.png)

### TEST 6

![alt text](vram_unitest_6.png)






1. test_llm_sleep_ack
   
description: 

Unit Test 1: LLM 信号握手与物理显存回收审计
验证 LLM Thinker 阶段在接收到睡眠指令后，能否准确返回物理显存回收的确定性数据（ACK）。
   
result:

--- Running test: test_llm_sleep_ack
INFO 02-19 13:26:36 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-19 13:26:36 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-19 13:26:36 [async_omni.py:885] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-19 13:26:36 [async_omni_llm.py:233] [AsyncOmniLLM] Engine Relay: Sleep Task 42204aa8-1f46-4e01-ada9-f8a229bd1027 (Level 2)
[0;36m(EngineCore_DP0 pid=128264)[0;0m [Stage-0] INFO 02-19 13:26:36 [base.py:143] [Omni Worker 0] Handshake Received: Task 42204aa8-1f46-4e01-ada9-f8a229bd1027, Level 2
[0;36m(EngineCore_DP0 pid=128264)[0;0m [Stage-0] INFO 02-19 13:26:37 [cumem.py:213] CuMemAllocator: sleep freed 9.49 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 9.49 GiB is discarded directly.
[0;36m(EngineCore_DP0 pid=128264)[0;0m [Stage-0] INFO 02-19 13:26:37 [base.py:125] [LLM Worker 0] Sleep mode freed 10.88 GiB.
[0;36m(EngineCore_DP0 pid=128264)[0;0m [Stage-0] INFO 02-19 13:26:37 [base.py:168] [Omni Worker 0] ACK emitted for Task 42204aa8-1f46-4e01-ada9-f8a229bd1027
[Stage-0] INFO 02-19 13:26:37 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-19 13:26:37 [async_omni.py:632] [AsyncOrchestrator] Intercepted wrapped ACK for task 42204aa8-1f46-4e01-ada9-f8a229bd1027 from stage-0
PASSEDGPU cleanup disabled
[rank0]:[W219 13:26:37.366831850 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank0]:[W219 13:26:37.417116291 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[Stage-0] INFO 02-19 13:26:37 [omni_stage.py:1428] Stage worker exiting
[Stage-1] INFO 02-19 13:26:38 [omni_stage.py:1428] Stage worker exiting


=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 1 passed, 6 deselected, 2 warnings in 130.71s (0:02:10) ============




2. test_diffusion_sleep_handshake
   
description: 

Unit Test 2: Diffusion 阶段分级睡眠动作与握手协议校验
物理触发 Diffusion 阶段的分级睡眠（Offload/Discard），验证分布式 Worker 在真实显存动作后的同步握手链路。
   

result:

--- Running test: test_diffusion_sleep_handshake
INFO 02-19 09:52:14 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-19 09:52:14 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-19 09:52:14 [async_omni.py:881] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-19 09:52:14 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: a0ce115b-0381-48ba-86c9-1eb92377877a (Level: 2)
[Stage-0] INFO 02-19 09:52:14 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task a0ce115b-0381-48ba-86c9-1eb92377877a (Level: 2)
[Stage-0] INFO 02-19 09:52:14 [diffusion_worker.py:253] [Diffusion Worker 0] Handshake Received: Task a0ce115b-0381-48ba-86c9-1eb92377877a, Level 2
[Stage-0] INFO 02-19 09:52:14 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-19 09:52:15 [diffusion_worker.py:209] Sleep mode freed 0.00 GiB memory, 14.89 GiB memory is still in use.
[Stage-0] INFO 02-19 09:52:15 [diffusion_worker.py:275] [Diffusion Worker 0]: ACK emitted. Freed 14.87 GiB.
[Stage-0] INFO 02-19 09:52:15 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-19 09:52:15 [async_omni.py:628] [AsyncOrchestrator] Intercepted wrapped ACK for task a0ce115b-0381-48ba-86c9-1eb92377877a from stage-0



3. test_cross_device_cleanup

description: 

Unit Test 3: 模型权重卸载与显存清空确定性审计
物理核验 Diffusion 模型权重在 Level1 或 level2模式下的卸载效率，确保显存物理占用达到预期阈值。
   
result:

--- Running test: test_cross_device_cleanup
INFO 02-18 10:10:42 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-18 10:10:42 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-18 10:10:42 [async_omni.py:881] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-18 10:10:42 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 0a3c7970-8920-4662-8e82-15523dcd6aed (Level: 2)
[Stage-0] INFO 02-18 10:10:42 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 0a3c7970-8920-4662-8e82-15523dcd6aed (Level: 2)
[Stage-0] INFO 02-18 10:10:42 [diffusion_worker.py:253] [Diffusion Worker 0] Handshake Received: Task 0a3c7970-8920-4662-8e82-15523dcd6aed, Level 2
[Stage-0] INFO 02-18 10:10:42 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-18 10:10:42 [diffusion_worker.py:209] Sleep mode freed 0.00 GiB memory, 14.89 GiB memory is still in use.
[Stage-0] INFO 02-18 10:10:42 [diffusion_worker.py:278] [Diffusion Worker 0]: ACK emitted. Freed 14.87 GiB.
[Stage-0] INFO 02-18 10:10:42 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-18 10:10:42 [async_omni.py:628] [AsyncOrchestrator] Intercepted wrapped ACK for task 0a3c7970-8920-4662-8e82-15523dcd6aed from stage-0




4. test_diffusion_integrity_bit_level
   
description:

Unit Test 4: 唤醒后推理一致性与位级精度校验
验证 Diffusion 模型在经历“运行-睡眠-唤醒”全生命周期后，生成的图像数据是否具备位级一致性，防止模型状态损坏。
   
result:

--- Running test: test_diffusion_integrity_bit_level
INFO 02-18 10:43:22 [async_omni.py:367] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-18 10:43:22 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-18 10:43:22 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-18 10:43:23 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-18 10:43:23 [diffusion_engine.py:110] Post-processing completed in 0.0489 seconds
INFO 02-18 10:43:23 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-18 10:43:23 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-18 10:43:23 [async_omni.py:881] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-18 10:43:23 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 1f8d0c78-b1e2-4109-9523-6f15eac6d206 (Level: 2)
[Stage-0] INFO 02-18 10:43:23 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 1f8d0c78-b1e2-4109-9523-6f15eac6d206 (Level: 2)
[Stage-0] INFO 02-18 10:43:23 [diffusion_worker.py:253] [Diffusion Worker 0] Handshake Received: Task 1f8d0c78-b1e2-4109-9523-6f15eac6d206, Level 2
[Stage-0] INFO 02-18 10:43:23 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-18 10:43:24 [diffusion_worker.py:209] Sleep mode freed 0.00 GiB memory, 14.89 GiB memory is still in use.
[Stage-0] INFO 02-18 10:43:24 [diffusion_worker.py:275] [Diffusion Worker 0]: ACK emitted. Freed 14.87 GiB.
[Stage-0] INFO 02-18 10:43:24 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-18 10:43:24 [async_omni.py:628] [AsyncOrchestrator] Intercepted wrapped ACK for task 1f8d0c78-b1e2-4109-9523-6f15eac6d206 from stage-0
INFO 02-18 10:43:24 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-18 10:43:24 [async_omni.py:903] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-18 10:43:24 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: 711290a9-5ae6-41b4-9be0-fdd5fd184a4c
[Stage-0] INFO 02-18 10:43:24 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task 711290a9-5ae6-41b4-9be0-fdd5fd184a4c to workers...
[Stage-0] INFO 02-18 10:43:24 [diffusion_worker.py:286] [Diffusion Worker 0] Responding to Wake-up Task: 711290a9-5ae6-41b4-9be0-fdd5fd184a4c
[Stage-0] INFO 02-18 10:43:24 [diffusion_worker.py:295] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-18 10:43:24 [async_omni.py:628] [AsyncOrchestrator] Intercepted wrapped ACK for task 711290a9-5ae6-41b4-9be0-fdd5fd184a4c from stage-0
INFO 02-18 10:43:24 [async_omni.py:367] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-18 10:43:24 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-18 10:43:24 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-18 10:43:24 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-18 10:43:25 [diffusion_engine.py:110] Post-processing completed in 0.0079 seconds




5. test_coordinated_cross_device
   
description: 

Unit Test 5: 异构引擎协同显存清理（共享 GPU测试）
模拟极端压力场景，测试 LLM Talker 与 Diffusion Base 协同运行在同一张 GPU 时，能否有序地完成显存物理释放与回收。
   
result:

--- Running test: test_coordinated_cross_device
INFO 02-20 10:34:11 [omni_stage.py:566] [Stage-1] Status transitioned to: TRANSITIONING
INFO 02-20 10:34:11 [omni_stage.py:570] [Stage-1] Submitting SLEEP task (Level: 2)
INFO 02-20 10:34:11 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
INFO 02-20 10:34:11 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-20 10:34:11 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-20 10:34:11 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-20 10:34:11 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: d26c3fa6-b43d-4f63-9c06-3d97b5c2bb44 (Level: 2)
[Stage-0] INFO 02-20 10:34:11 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task d26c3fa6-b43d-4f63-9c06-3d97b5c2bb44 (Level: 2)
[Stage-0] INFO 02-20 10:34:11 [diffusion_worker.py:253] [Diffusion Worker 0] Handshake Received: Task d26c3fa6-b43d-4f63-9c06-3d97b5c2bb44, Level 2
[Stage-1] INFO 02-20 10:34:11 [async_omni_llm.py:228] [AsyncOmniLLM] Engine Relay: Sleep Task 35f6c7c9-4168-42c0-81a8-49bdd1a39d88 (Level 2)
[0;36m(EngineCore_DP0 pid=140115)[0;0m [Stage-1] INFO 02-20 10:34:11 [base.py:157] [LLM Worker 0] Handshake Received: Task 35f6c7c9-4168-42c0-81a8-49bdd1a39d88, Level 2
[0;36m(EngineCore_DP0 pid=140115)[0;0m [Stage-1] INFO 02-20 10:34:11 [cumem.py:213] CuMemAllocator: sleep freed 0.47 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 0.47 GiB is discarded directly.
[0;36m(EngineCore_DP0 pid=140115)[0;0m [Stage-1] INFO 02-20 10:34:11 [base.py:126] [LLM Worker 0] Level 2 Sleep: Offloading weights to CPU...
[Stage-0] INFO 02-20 10:34:11 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-20 10:34:11 [diffusion_worker.py:209] Sleep mode freed 0.00 GiB memory, 14.89 GiB memory is still in use.
[Stage-0] INFO 02-20 10:34:11 [diffusion_worker.py:275] [Diffusion Worker 0]: ACK emitted. Freed 14.87 GiB.
[Stage-0] INFO 02-20 10:34:11 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-20 10:34:11 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task d26c3fa6-b43d-4f63-9c06-3d97b5c2bb44 from stage-0
INFO 02-20 10:34:11 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
[0;36m(EngineCore_DP0 pid=140115)[0;0m [Stage-1] INFO 02-20 10:34:13 [base.py:134] [LLM Worker 0] Sleep mode freed 4.53 GiB.
[0;36m(EngineCore_DP0 pid=140115)[0;0m [Stage-1] INFO 02-20 10:34:13 [base.py:182] [LLM Worker 0] ACK emitted for Task 35f6c7c9-4168-42c0-81a8-49bdd1a39d88
[Stage-1] INFO 02-20 10:34:13 [omni_stage.py:1323] [Stage-1] Sleep ACKs forwarded to Orchestrator
INFO 02-20 10:34:13 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 35f6c7c9-4168-42c0-81a8-49bdd1a39d88 from stage-1
INFO 02-20 10:34:13 [omni_stage.py:566] [Stage-1] Status transitioned to: SLEEPING
PASSEDGPU cleanup disabled
[Stage-0] INFO 02-20 10:34:14 [diffusion_worker.py:436] Worker 0: Received shutdown message
[Stage-0] INFO 02-20 10:34:14 [diffusion_worker.py:457] event loop terminated.
[Stage-0] INFO 02-20 10:34:14 [diffusion_worker.py:488] Worker 0: Shutdown complete.
[Stage-0] INFO 02-20 10:34:18 [async_omni_diffusion.py:213] AsyncOmniDiffusion closed
[Stage-0] INFO 02-20 10:34:18 [omni_stage.py:1428] Stage worker exiting
[rank0]:[W220 10:34:19.356637799 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[rank0]:[W220 10:34:19.383553223 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
[Stage-0] INFO 02-20 10:34:20 [omni_stage.py:1428] Stage worker exiting
[Stage-1] INFO 02-20 10:34:20 [omni_stage.py:1428] Stage worker exiting


=============================== warnings summary ===============================
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========== 1 passed, 4 deselected, 2 warnings in 384.72s (0:06:24) ============



6. test_diffusion_vram_lifecycle_audit 

description: 

Unit Test 6: 端到端显存生命周期全循环审计
完整覆盖“活跃 (Active) -> 深度睡眠 (Deep Sleep) -> 重新唤醒 (Restored)”的闭环流程，验证显存池的动态吞吐能力。
   
result:

--- Running test: test_diffusion_vram_lifecycle_audit
INFO 02-22 05:31:24 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-22 05:31:24 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-22 05:31:24 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-22 05:31:24 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: cb09dca7-2a3d-46ad-b47c-6b66243b9680 (Level: 2)
[Stage-0] INFO 02-22 05:31:24 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task cb09dca7-2a3d-46ad-b47c-6b66243b9680 (Level: 2)
[Stage-0] INFO 02-22 05:31:24 [diffusion_worker.py:253] [Diffusion Worker 0] Handshake Received: Task cb09dca7-2a3d-46ad-b47c-6b66243b9680, Level 2
[Stage-0] INFO 02-22 05:31:24 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-22 05:31:25 [diffusion_worker.py:209] Sleep mode freed 0.00 GiB memory, 14.89 GiB memory is still in use.
[Stage-0] INFO 02-22 05:31:25 [diffusion_worker.py:275] [Diffusion Worker 0]: ACK emitted. Freed 14.87 GiB.
[Stage-0] INFO 02-22 05:31:25 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-22 05:31:25 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task cb09dca7-2a3d-46ad-b47c-6b66243b9680 from stage-0
INFO 02-22 05:31:25 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
INFO 02-22 05:31:26 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-22 05:31:26 [async_omni.py:921] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-22 05:31:26 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: 6dddc7c3-b77c-4c28-92aa-6eb739f9f704
[Stage-0] INFO 02-22 05:31:26 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task 6dddc7c3-b77c-4c28-92aa-6eb739f9f704 to workers...
[Stage-0] INFO 02-22 05:31:26 [diffusion_worker.py:286] [Diffusion Worker 0] Responding to Wake-up Task: 6dddc7c3-b77c-4c28-92aa-6eb739f9f704
[Stage-0] INFO 02-22 05:31:26 [diffusion_worker.py:295] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-22 05:31:26 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 6dddc7c3-b77c-4c28-92aa-6eb739f9f704 from stage-0
INFO 02-22 05:31:26 [omni_stage.py:566] [Stage-0] Status transitioned to: RUNNING
INFO 02-22 05:31:27 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-22 05:31:27 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-22 05:31:27 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-22 05:31:27 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-22 05:31:27 [diffusion_engine.py:110] Post-processing completed in 0.0516 seconds







## AMD

MI300X TP=2




1. test_llm_sleep_ack
   
description: 
   
result:





2. test_diffusion_sleep_handshake
   
description: 
   
result:

#### level 1

--- Running test: test_diffusion_sleep_handshake
INFO 02-23 06:03:20 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 06:03:20 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-23 06:03:20 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:03:20 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 683740e3-53fe-4846-abb9-00afc9c4bfcf (Level: 1)
[Stage-0] INFO 02-23 06:03:20 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 683740e3-53fe-4846-abb9-00afc9c4bfcf (Level: 1)
[Stage-0] INFO 02-23 06:03:20 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 683740e3-53fe-4846-abb9-00afc9c4bfcf, Level 1
[Stage-0] INFO 02-23 06:03:23 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 14.88 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-23 06:03:23 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 15.90 GiB. 2.06GiB memory is still in use.
[Stage-0] INFO 02-23 06:03:23 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.90 GiB.
[Stage-0] INFO 02-23 06:03:23 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 06:03:23 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 683740e3-53fe-4846-abb9-00afc9c4bfcf from stage-0
INFO 02-23 06:03:23 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING



#### level 2
--- Running test: test_diffusion_sleep_handshake
INFO 02-23 05:58:49 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 05:58:49 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-23 05:58:49 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 05:58:49 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: c8e254b5-cfa9-4130-8d24-2bbfa5fdc15b (Level: 2)
[Stage-0] INFO 02-23 05:58:49 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task c8e254b5-cfa9-4130-8d24-2bbfa5fdc15b (Level: 2)
[Stage-0] INFO 02-23 05:58:49 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task c8e254b5-cfa9-4130-8d24-2bbfa5fdc15b, Level 2
[Stage-0] INFO 02-23 05:58:49 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.88 GiB is discarded directly.
[Stage-0] INFO 02-23 05:58:49 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 15.90 GiB. 2.06GiB memory is still in use.
[Stage-0] INFO 02-23 05:58:49 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.90 GiB.
[Stage-0] INFO 02-23 05:58:49 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 05:58:49 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task c8e254b5-cfa9-4130-8d24-2bbfa5fdc15b from stage-0
INFO 02-23 05:58:49 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING


3. test_cross_device_cleanup

description: 
   
result:

#### level 3
--- Running test: test_cross_device_cleanup
INFO 02-23 06:17:00 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 06:17:00 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-23 06:17:00 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:17:00 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 4103d96e-3f7d-44a6-97b8-3af4562be6f9 (Level: 1)
[Stage-0] INFO 02-23 06:17:00 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 4103d96e-3f7d-44a6-97b8-3af4562be6f9 (Level: 1)
[Stage-0] INFO 02-23 06:17:00 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 4103d96e-3f7d-44a6-97b8-3af4562be6f9, Level 1
[Stage-0] INFO 02-23 06:17:02 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 14.88 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-23 06:17:03 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 15.90 GiB. 2.06GiB memory is still in use.
[Stage-0] INFO 02-23 06:17:03 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.90 GiB.
[Stage-0] INFO 02-23 06:17:03 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 06:17:03 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 4103d96e-3f7d-44a6-97b8-3af4562be6f9 from stage-0
INFO 02-23 06:17:03 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING




#### level 2
--- Running test: test_cross_device_cleanup
INFO 02-23 06:13:56 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 06:13:56 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-23 06:13:56 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:13:56 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 5921c52d-c8c2-4736-b01c-ecc7d5b70eb6 (Level: 2)
[Stage-0] INFO 02-23 06:13:56 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 5921c52d-c8c2-4736-b01c-ecc7d5b70eb6 (Level: 2)
[Stage-0] INFO 02-23 06:13:56 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 5921c52d-c8c2-4736-b01c-ecc7d5b70eb6, Level 2
[Stage-0] INFO 02-23 06:13:56 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.88 GiB is discarded directly.
[Stage-0] INFO 02-23 06:13:57 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 15.90 GiB. 2.06GiB memory is still in use.
[Stage-0] INFO 02-23 06:13:57 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.90 GiB.
[Stage-0] INFO 02-23 06:13:57 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 06:13:57 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 5921c52d-c8c2-4736-b01c-ecc7d5b70eb6 from stage-0
INFO 02-23 06:13:57 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING



4. test_diffusion_integrity_bit_level
   
description: 
   
result:

#### level 1
--- Running test: test_diffusion_integrity_bit_level
INFO 02-23 06:24:45 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-23 06:24:45 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-23 06:24:45 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-23 06:24:46 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-23 06:24:46 [diffusion_engine.py:110] Post-processing completed in 0.1148 seconds
INFO 02-23 06:24:46 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 06:24:46 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-23 06:24:46 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:24:46 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 352e7bd2-4d38-41e6-a4b2-8cdfb0a16394 (Level: 1)
[Stage-0] INFO 02-23 06:24:46 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 352e7bd2-4d38-41e6-a4b2-8cdfb0a16394 (Level: 1)
[Stage-0] INFO 02-23 06:24:46 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 352e7bd2-4d38-41e6-a4b2-8cdfb0a16394, Level 1
[Stage-0] INFO 02-23 06:24:48 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 14.88 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-23 06:24:49 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 15.00 GiB. 2.08GiB memory is still in use.
[Stage-0] INFO 02-23 06:24:49 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.00 GiB.
[Stage-0] INFO 02-23 06:24:49 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 06:24:49 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 352e7bd2-4d38-41e6-a4b2-8cdfb0a16394 from stage-0
INFO 02-23 06:24:49 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
INFO 02-23 06:24:49 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-23 06:24:49 [async_omni.py:921] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:24:49 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: 771a17a8-6698-4ebf-9904-07f84a9a0fa1
[Stage-0] INFO 02-23 06:24:49 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task 771a17a8-6698-4ebf-9904-07f84a9a0fa1 to workers...
[Stage-0] INFO 02-23 06:24:49 [diffusion_worker.py:257] [Diffusion Worker 0] Responding to Wake-up Task: 771a17a8-6698-4ebf-9904-07f84a9a0fa1
[Stage-0] INFO 02-23 06:24:49 [diffusion_worker.py:217] [Diffusion Worker 0] Wake-up complete.
[Stage-0] INFO 02-23 06:24:49 [diffusion_worker.py:266] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-23 06:24:49 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 771a17a8-6698-4ebf-9904-07f84a9a0fa1 from stage-0
INFO 02-23 06:24:49 [omni_stage.py:566] [Stage-0] Status transitioned to: RUNNING
INFO 02-23 06:24:49 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-23 06:24:49 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-23 06:24:49 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-23 06:24:49 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-23 06:24:49 [diffusion_engine.py:110] Post-processing completed in 0.0154 seconds




#### level 2
--- Running test: test_diffusion_integrity_bit_level
INFO 02-23 06:20:03 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-23 06:20:03 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-23 06:20:03 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-23 06:20:03 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-23 06:20:03 [diffusion_engine.py:110] Post-processing completed in 0.1070 seconds
INFO 02-23 06:20:03 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 06:20:03 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-23 06:20:03 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:20:03 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 749aae02-5c8e-420a-a655-d3601cc73f48 (Level: 2)
[Stage-0] INFO 02-23 06:20:03 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 749aae02-5c8e-420a-a655-d3601cc73f48 (Level: 2)
[Stage-0] INFO 02-23 06:20:03 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 749aae02-5c8e-420a-a655-d3601cc73f48, Level 2
[Stage-0] INFO 02-23 06:20:04 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.88 GiB is discarded directly.
[Stage-0] INFO 02-23 06:20:04 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 15.00 GiB. 2.08GiB memory is still in use.
[Stage-0] INFO 02-23 06:20:04 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.00 GiB.
[Stage-0] INFO 02-23 06:20:04 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 06:20:04 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 749aae02-5c8e-420a-a655-d3601cc73f48 from stage-0
INFO 02-23 06:20:04 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
INFO 02-23 06:20:04 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-23 06:20:04 [async_omni.py:921] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:20:04 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: bec85424-de88-4da8-9c33-91fedf274965
[Stage-0] INFO 02-23 06:20:04 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task bec85424-de88-4da8-9c33-91fedf274965 to workers...
[Stage-0] INFO 02-23 06:20:04 [diffusion_worker.py:257] [Diffusion Worker 0] Responding to Wake-up Task: bec85424-de88-4da8-9c33-91fedf274965
[Stage-0] INFO 02-23 06:20:04 [diffusion_worker.py:217] [Diffusion Worker 0] Wake-up complete.
[Stage-0] INFO 02-23 06:20:04 [diffusion_worker.py:266] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-23 06:20:04 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task bec85424-de88-4da8-9c33-91fedf274965 from stage-0
INFO 02-23 06:20:04 [omni_stage.py:566] [Stage-0] Status transitioned to: RUNNING
INFO 02-23 06:20:04 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-23 06:20:04 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-23 06:20:04 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-23 06:20:04 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-23 06:20:04 [diffusion_engine.py:110] Post-processing completed in 0.0099 seconds


5. test_coordinated_cross_device
   
description: 
   
result:







6. test_diffusion_vram_lifecycle_audit 

description: 
   
result:


#### level 1
--- Running test: test_diffusion_vram_lifecycle_audit
INFO 02-23 06:29:36 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 06:29:36 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-23 06:29:36 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:29:36 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 38de3321-1f5b-48a8-acb9-6f01d03e6b42 (Level: 1)
[Stage-0] INFO 02-23 06:29:36 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 38de3321-1f5b-48a8-acb9-6f01d03e6b42 (Level: 1)
[Stage-0] INFO 02-23 06:29:36 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 38de3321-1f5b-48a8-acb9-6f01d03e6b42, Level 1
[Stage-0] INFO 02-23 06:29:39 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 14.88 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-23 06:29:39 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 15.90 GiB. 2.06GiB memory is still in use.
[Stage-0] INFO 02-23 06:29:39 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.90 GiB.
[Stage-0] INFO 02-23 06:29:39 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 06:29:39 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 38de3321-1f5b-48a8-acb9-6f01d03e6b42 from stage-0
INFO 02-23 06:29:39 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
INFO 02-23 06:29:40 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-23 06:29:40 [async_omni.py:921] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:29:40 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: c883f23f-4713-4bfd-bbbb-c44657930363
[Stage-0] INFO 02-23 06:29:40 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task c883f23f-4713-4bfd-bbbb-c44657930363 to workers...
[Stage-0] INFO 02-23 06:29:40 [diffusion_worker.py:257] [Diffusion Worker 0] Responding to Wake-up Task: c883f23f-4713-4bfd-bbbb-c44657930363
[Stage-0] INFO 02-23 06:29:40 [diffusion_worker.py:217] [Diffusion Worker 0] Wake-up complete.
[Stage-0] INFO 02-23 06:29:40 [diffusion_worker.py:266] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-23 06:29:40 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task c883f23f-4713-4bfd-bbbb-c44657930363 from stage-0
INFO 02-23 06:29:40 [omni_stage.py:566] [Stage-0] Status transitioned to: RUNNING
INFO 02-23 06:29:41 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-23 06:29:41 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-23 06:29:41 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-23 06:29:41 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-23 06:29:42 [diffusion_engine.py:110] Post-processing completed in 0.1179 seconds



#### level 2
--- Running test: test_diffusion_vram_lifecycle_audit
INFO 02-23 06:27:02 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-23 06:27:02 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-23 06:27:02 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:27:02 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 29e1f28f-0352-41a8-9907-47a3fbf0f0d9 (Level: 2)
[Stage-0] INFO 02-23 06:27:02 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 29e1f28f-0352-41a8-9907-47a3fbf0f0d9 (Level: 2)
[Stage-0] INFO 02-23 06:27:02 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 29e1f28f-0352-41a8-9907-47a3fbf0f0d9, Level 2
[Stage-0] INFO 02-23 06:27:02 [cumem.py:213] CuMemAllocator: sleep freed 14.88 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.88 GiB is discarded directly.
[Stage-0] INFO 02-23 06:27:03 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 15.90 GiB. 2.06GiB memory is still in use.
[Stage-0] INFO 02-23 06:27:03 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 15.90 GiB.
[Stage-0] INFO 02-23 06:27:03 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-23 06:27:03 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 29e1f28f-0352-41a8-9907-47a3fbf0f0d9 from stage-0
INFO 02-23 06:27:03 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
INFO 02-23 06:27:04 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-23 06:27:04 [async_omni.py:921] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-23 06:27:04 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: 388daadd-f527-4f6f-b722-a8215e63b298
[Stage-0] INFO 02-23 06:27:04 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task 388daadd-f527-4f6f-b722-a8215e63b298 to workers...
[Stage-0] INFO 02-23 06:27:04 [diffusion_worker.py:257] [Diffusion Worker 0] Responding to Wake-up Task: 388daadd-f527-4f6f-b722-a8215e63b298
[Stage-0] INFO 02-23 06:27:04 [diffusion_worker.py:217] [Diffusion Worker 0] Wake-up complete.
[Stage-0] INFO 02-23 06:27:04 [diffusion_worker.py:266] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-23 06:27:04 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 388daadd-f527-4f6f-b722-a8215e63b298 from stage-0
INFO 02-23 06:27:04 [omni_stage.py:566] [Stage-0] Status transitioned to: RUNNING
INFO 02-23 06:27:05 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-23 06:27:05 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-23 06:27:05 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-23 06:27:05 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-23 06:27:05 [diffusion_engine.py:110] Post-processing completed in 0.1101 seconds