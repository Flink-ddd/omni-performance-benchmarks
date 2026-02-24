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

1. test AMD, pending.
2. test NVIDIA, pending. 












## NVIDIA 

NVIDIA RTXA6000 TP=2


### TEST 5

![alt text](vram_recycling_unittest_5.png)

### TEST 6

![alt text](vram_unitest_6.png)






1. test_llm_sleep_ack
   
description: 

Unit Test 1: LLM 信号握手与物理显存回收审计
验证 LLM Thinker 阶段在接收到睡眠指令后，能否准确返回物理显存回收的确定性数据（ACK）。
   
result:

#### level 1

--- Running test: test_llm_sleep_ack
INFO 02-24 03:23:10 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:23:10 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-24 03:23:10 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:23:10 [async_omni_llm.py:229] [AsyncOmniLLM] Engine Relay: Sleep Task cb1f9bf0-a3e1-4f9b-8dc8-000587e4a660 (Level 1)
[0;36m(EngineCore_DP0 pid=30770)[0;0m [Stage-0] INFO 02-24 03:23:10 [base.py:166] [LLM Worker 0] Handshake Received: Task cb1f9bf0-a3e1-4f9b-8dc8-000587e4a660, Level 1
[0;36m(EngineCore_DP0 pid=30770)[0;0m [Stage-0] INFO 02-24 03:23:19 [cumem.py:213] CuMemAllocator: sleep freed 42.44 GiB memory in total, of which 9.11 GiB is backed up in CPU and the rest 33.33 GiB is discarded directly.
[0;36m(EngineCore_DP0 pid=30770)[0;0m [Stage-0] INFO 02-24 03:23:19 [base.py:148] [LLM Worker 0] Level 1 Sleep: Freed 43.80 GiB. 0.66GiB memory is still in use.
[0;36m(EngineCore_DP0 pid=30770)[0;0m [Stage-0] INFO 02-24 03:23:19 [base.py:188] [LLM Worker 0] ACK emitted for Task cb1f9bf0-a3e1-4f9b-8dc8-000587e4a660
[Stage-0] INFO 02-24 03:23:19 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:23:19 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task cb1f9bf0-a3e1-4f9b-8dc8-000587e4a660 from stage-0
INFO 02-24 03:23:19 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING


#### level 2

--- Running test: test_llm_sleep_ack
INFO 02-24 03:26:20 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:26:20 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-24 03:26:20 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:26:20 [async_omni_llm.py:229] [AsyncOmniLLM] Engine Relay: Sleep Task f5259cd8-fe3a-408b-85f8-15631d2038ad (Level 2)
[0;36m(EngineCore_DP0 pid=33846)[0;0m [Stage-0] INFO 02-24 03:26:20 [base.py:166] [LLM Worker 0] Handshake Received: Task f5259cd8-fe3a-408b-85f8-15631d2038ad, Level 2
[0;36m(EngineCore_DP0 pid=33846)[0;0m [Stage-0] INFO 02-24 03:26:20 [cumem.py:213] CuMemAllocator: sleep freed 42.44 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 42.44 GiB is discarded directly.
[0;36m(EngineCore_DP0 pid=33846)[0;0m [Stage-0] INFO 02-24 03:26:20 [base.py:148] [LLM Worker 0] Level 2 Sleep: Freed 43.83 GiB. 0.64GiB memory is still in use.
[0;36m(EngineCore_DP0 pid=33846)[0;0m [Stage-0] INFO 02-24 03:26:20 [base.py:188] [LLM Worker 0] ACK emitted for Task f5259cd8-fe3a-408b-85f8-15631d2038ad
[Stage-0] INFO 02-24 03:26:20 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:26:20 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task f5259cd8-fe3a-408b-85f8-15631d2038ad from stage-0
INFO 02-24 03:26:20 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING




2. test_diffusion_sleep_handshake
   
description: 

Unit Test 2: Diffusion 阶段分级睡眠动作与握手协议校验
物理触发 Diffusion 阶段的分级睡眠（Offload/Discard），验证分布式 Worker 在真实显存动作后的同步握手链路。
   

result:

#### level 1

--- Running test: test_diffusion_sleep_handshake
INFO 02-24 03:28:19 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:28:19 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-24 03:28:19 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:28:19 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: a42a1cbf-8d2d-49c5-bd27-711b74ab6c68 (Level: 1)
[Stage-0] INFO 02-24 03:28:19 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task a42a1cbf-8d2d-49c5-bd27-711b74ab6c68 (Level: 1)
[Stage-0] INFO 02-24 03:28:19 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task a42a1cbf-8d2d-49c5-bd27-711b74ab6c68, Level 1
[Stage-0] INFO 02-24 03:28:31 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 14.89 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-24 03:28:32 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 18.37 GiB. 0.41GiB memory is still in use.
[Stage-0] INFO 02-24 03:28:32 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.37 GiB.
[Stage-0] INFO 02-24 03:28:32 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:28:32 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task a42a1cbf-8d2d-49c5-bd27-711b74ab6c68 from stage-0
INFO 02-24 03:28:32 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING


#### level 2

--- Running test: test_diffusion_sleep_handshake
INFO 02-24 03:30:13 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:30:13 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-24 03:30:13 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:30:13 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: ac2c59b3-b4a6-4321-9c4e-37d552d15a24 (Level: 2)
[Stage-0] INFO 02-24 03:30:13 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task ac2c59b3-b4a6-4321-9c4e-37d552d15a24 (Level: 2)
[Stage-0] INFO 02-24 03:30:13 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task ac2c59b3-b4a6-4321-9c4e-37d552d15a24, Level 2
[Stage-0] INFO 02-24 03:30:13 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-24 03:30:13 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 18.40 GiB. 0.37GiB memory is still in use.
[Stage-0] INFO 02-24 03:30:13 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.40 GiB.
[Stage-0] INFO 02-24 03:30:13 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:30:13 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task ac2c59b3-b4a6-4321-9c4e-37d552d15a24 from stage-0
INFO 02-24 03:30:13 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING




3. test_cross_device_cleanup

description: 

Unit Test 3: 模型权重卸载与显存清空确定性审计
物理核验 Diffusion 模型权重在 Level1 或 level2模式下的卸载效率，确保显存物理占用达到预期阈值。
   

result:

#### level 1

--- Running test: test_cross_device_cleanup
INFO 02-24 03:32:04 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:32:04 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-24 03:32:04 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:32:04 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 144b63c4-d024-46a1-aa86-f8259b12f4ba (Level: 1)
[Stage-0] INFO 02-24 03:32:04 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 144b63c4-d024-46a1-aa86-f8259b12f4ba (Level: 1)
[Stage-0] INFO 02-24 03:32:04 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 144b63c4-d024-46a1-aa86-f8259b12f4ba, Level 1
[Stage-0] INFO 02-24 03:32:19 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 14.89 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-24 03:32:20 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 18.37 GiB. 0.41GiB memory is still in use.
[Stage-0] INFO 02-24 03:32:20 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.37 GiB.
[Stage-0] INFO 02-24 03:32:20 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:32:20 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 144b63c4-d024-46a1-aa86-f8259b12f4ba from stage-0
INFO 02-24 03:32:20 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING



#### level 2

--- Running test: test_cross_device_cleanup
INFO 02-24 03:33:48 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:33:48 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-24 03:33:48 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:33:48 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 88f4c69c-31a8-42e7-b23b-0949ef9831aa (Level: 2)
[Stage-0] INFO 02-24 03:33:48 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 88f4c69c-31a8-42e7-b23b-0949ef9831aa (Level: 2)
[Stage-0] INFO 02-24 03:33:48 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 88f4c69c-31a8-42e7-b23b-0949ef9831aa, Level 2
[Stage-0] INFO 02-24 03:33:48 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-24 03:33:49 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 18.40 GiB. 0.37GiB memory is still in use.
[Stage-0] INFO 02-24 03:33:49 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.40 GiB.
[Stage-0] INFO 02-24 03:33:49 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:33:49 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 88f4c69c-31a8-42e7-b23b-0949ef9831aa from stage-0
INFO 02-24 03:33:49 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING


4. test_diffusion_integrity_bit_level
   
description:

Unit Test 4: 唤醒后推理一致性与位级精度校验
验证 Diffusion 模型在经历“运行-睡眠-唤醒”全生命周期后，生成的图像数据是否具备位级一致性，防止模型状态损坏。
   
result:

#### level 1

--- Running test: test_diffusion_integrity_bit_level
INFO 02-24 03:38:51 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-24 03:38:51 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-24 03:38:51 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-24 03:38:52 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-24 03:38:52 [diffusion_engine.py:110] Post-processing completed in 0.0564 seconds
INFO 02-24 03:38:52 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:38:52 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-24 03:38:52 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:38:52 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 133b2334-0c15-401e-8f6b-a0d6d16f35ca (Level: 1)
[Stage-0] INFO 02-24 03:38:52 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 133b2334-0c15-401e-8f6b-a0d6d16f35ca (Level: 1)
[Stage-0] INFO 02-24 03:38:52 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task 133b2334-0c15-401e-8f6b-a0d6d16f35ca, Level 1
[Stage-0] INFO 02-24 03:39:04 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 14.89 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-24 03:39:04 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 18.38 GiB. 0.39GiB memory is still in use.
[Stage-0] INFO 02-24 03:39:04 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.38 GiB.
[Stage-0] INFO 02-24 03:39:04 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:39:04 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 133b2334-0c15-401e-8f6b-a0d6d16f35ca from stage-0
INFO 02-24 03:39:04 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
INFO 02-24 03:39:04 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-24 03:39:04 [async_omni.py:921] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:39:04 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: f4e43980-b87d-4220-91fc-b252ec69bca3
[Stage-0] INFO 02-24 03:39:04 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task f4e43980-b87d-4220-91fc-b252ec69bca3 to workers...
[Stage-0] INFO 02-24 03:39:04 [diffusion_worker.py:257] [Diffusion Worker 0] Responding to Wake-up Task: f4e43980-b87d-4220-91fc-b252ec69bca3
[Stage-0] INFO 02-24 03:39:05 [diffusion_worker.py:217] [Diffusion Worker 0] Wake-up complete.
[Stage-0] INFO 02-24 03:39:05 [diffusion_worker.py:266] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-24 03:39:05 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task f4e43980-b87d-4220-91fc-b252ec69bca3 from stage-0
INFO 02-24 03:39:05 [omni_stage.py:566] [Stage-0] Status transitioned to: RUNNING
INFO 02-24 03:39:05 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-24 03:39:05 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-24 03:39:05 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-24 03:39:05 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-24 03:39:05 [diffusion_engine.py:110] Post-processing completed in 0.0122 seconds




#### level 2

--- Running test: test_diffusion_integrity_bit_level
INFO 02-24 03:40:56 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-24 03:40:56 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-24 03:40:56 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-24 03:40:57 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-24 03:40:57 [diffusion_engine.py:110] Post-processing completed in 0.0552 seconds
INFO 02-24 03:40:57 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:40:57 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-24 03:40:57 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:40:57 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: ed5b5e3b-217b-474a-a95e-1a3bce031a92 (Level: 2)
[Stage-0] INFO 02-24 03:40:57 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task ed5b5e3b-217b-474a-a95e-1a3bce031a92 (Level: 2)
[Stage-0] INFO 02-24 03:40:57 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task ed5b5e3b-217b-474a-a95e-1a3bce031a92, Level 2
[Stage-0] INFO 02-24 03:40:57 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-24 03:40:57 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 18.42 GiB. 0.35GiB memory is still in use.
[Stage-0] INFO 02-24 03:40:57 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.42 GiB.
[Stage-0] INFO 02-24 03:40:57 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:40:57 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task ed5b5e3b-217b-474a-a95e-1a3bce031a92 from stage-0
INFO 02-24 03:40:57 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING
INFO 02-24 03:40:57 [omni_stage.py:579] [Stage-0] Submitting WAKE_UP task
INFO 02-24 03:40:57 [async_omni.py:921] [AsyncOrchestrator] Wake-up initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:40:57 [async_omni_diffusion.py:318] [Entrypoint] Relaying WakeUp Task: 42e65f26-3fb1-47ff-8df3-cc0ed8a6381c
[Stage-0] INFO 02-24 03:40:57 [diffusion_engine.py:424] [Diffusion Engine Relay] Dispatching Wake-up Task 42e65f26-3fb1-47ff-8df3-cc0ed8a6381c to workers...
[Stage-0] INFO 02-24 03:40:57 [diffusion_worker.py:257] [Diffusion Worker 0] Responding to Wake-up Task: 42e65f26-3fb1-47ff-8df3-cc0ed8a6381c
[Stage-0] INFO 02-24 03:40:57 [diffusion_worker.py:217] [Diffusion Worker 0] Wake-up complete.
[Stage-0] INFO 02-24 03:40:57 [diffusion_worker.py:266] [Diffusion Worker 0] Wake-up confirmed.
INFO 02-24 03:40:57 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 42e65f26-3fb1-47ff-8df3-cc0ed8a6381c from stage-0
INFO 02-24 03:40:57 [omni_stage.py:566] [Stage-0] Status transitioned to: RUNNING
INFO 02-24 03:40:57 [async_omni.py:378] [AsyncOrchestrator] Entering scheduling loop: stages=1, final_stage=0
[Stage-0] INFO 02-24 03:40:57 [manager.py:538] Deactivating all adapters: 0 layers
[Stage-0] WARNING 02-24 03:40:57 [kv_transfer_manager.py:356] No connector available for receiving KV cache
[Stage-0] INFO 02-24 03:40:58 [diffusion_engine.py:92] Generation completed successfully.
[Stage-0] INFO 02-24 03:40:58 [diffusion_engine.py:110] Post-processing completed in 0.0086 seconds



5. test_coordinated_cross_device
   
description: 

Unit Test 5: 异构引擎协同显存清理（共享 GPU测试）
模拟极端压力场景，测试 LLM Talker 与 Diffusion Base 协同运行在同一张 GPU 时，能否有序地完成显存物理释放与回收。
   
result:

#### level 1

--- Running test: test_coordinated_cross_device
INFO 02-24 03:48:47 [omni_stage.py:566] [Stage-1] Status transitioned to: TRANSITIONING
INFO 02-24 03:48:47 [omni_stage.py:570] [Stage-1] Submitting SLEEP task (Level: 1)
INFO 02-24 03:48:47 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-1] INFO 02-24 03:48:47 [async_omni_llm.py:229] [AsyncOmniLLM] Engine Relay: Sleep Task 5b92ca6b-f64d-44cf-9d6a-03b81d98612c (Level 1)
[0;36m(EngineCore_DP0 pid=47422)[0;0m [Stage-1] INFO 02-24 03:48:47 [base.py:166] [LLM Worker 0] Handshake Received: Task 5b92ca6b-f64d-44cf-9d6a-03b81d98612c, Level 1
[0;36m(EngineCore_DP0 pid=47422)[0;0m [Stage-1] INFO 02-24 03:48:51 [cumem.py:213] CuMemAllocator: sleep freed 13.52 GiB memory in total, of which 4.01 GiB is backed up in CPU and the rest 9.52 GiB is discarded directly.
[0;36m(EngineCore_DP0 pid=47422)[0;0m [Stage-1] INFO 02-24 03:48:51 [base.py:148] [LLM Worker 0] Level 1 Sleep: Freed 13.60 GiB. 19.42GiB memory is still in use.
[0;36m(EngineCore_DP0 pid=47422)[0;0m [Stage-1] INFO 02-24 03:48:51 [base.py:188] [LLM Worker 0] ACK emitted for Task 5b92ca6b-f64d-44cf-9d6a-03b81d98612c
[Stage-1] INFO 02-24 03:48:51 [omni_stage.py:1323] [Stage-1] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:48:51 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 5b92ca6b-f64d-44cf-9d6a-03b81d98612c from stage-1
INFO 02-24 03:48:51 [omni_stage.py:566] [Stage-1] Status transitioned to: SLEEPING
INFO 02-24 03:48:52 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 03:48:52 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 1)
INFO 02-24 03:48:52 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 03:48:52 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: f5d21d44-f170-4996-ac12-508c71a9990d (Level: 1)
[Stage-0] INFO 02-24 03:48:52 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task f5d21d44-f170-4996-ac12-508c71a9990d (Level: 1)
[Stage-0] INFO 02-24 03:48:52 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task f5d21d44-f170-4996-ac12-508c71a9990d, Level 1
[Stage-0] INFO 02-24 03:49:04 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 14.89 GiB is backed up in CPU and the rest 0.00 GiB is discarded directly.
[Stage-0] INFO 02-24 03:49:05 [diffusion_worker.py:198] [Diffusion Worker 0] Level 1 Sleep: Freed 18.37 GiB. 1.05GiB memory is still in use.
[Stage-0] INFO 02-24 03:49:05 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.37 GiB.
[Stage-0] INFO 02-24 03:49:05 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 03:49:05 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task f5d21d44-f170-4996-ac12-508c71a9990d from stage-0
INFO 02-24 03:49:05 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING



#### level 2

--- Running test: test_coordinated_cross_device
INFO 02-24 05:17:40 [omni_stage.py:566] [Stage-1] Status transitioned to: TRANSITIONING
INFO 02-24 05:17:40 [omni_stage.py:570] [Stage-1] Submitting SLEEP task (Level: 2)
INFO 02-24 05:17:40 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-1] INFO 02-24 05:17:40 [async_omni_llm.py:229] [AsyncOmniLLM] Engine Relay: Sleep Task 5e836a82-39cf-4e2e-a167-c803b1c8ace0 (Level 2)
[0;36m(EngineCore_DP0 pid=59255)[0;0m [Stage-1] INFO 02-24 05:17:40 [base.py:166] [LLM Worker 0] Handshake Received: Task 5e836a82-39cf-4e2e-a167-c803b1c8ace0, Level 2
[0;36m(EngineCore_DP0 pid=59255)[0;0m [Stage-1] INFO 02-24 05:17:40 [cumem.py:213] CuMemAllocator: sleep freed 13.52 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 13.52 GiB is discarded directly.
[0;36m(EngineCore_DP0 pid=59255)[0;0m [Stage-1] INFO 02-24 05:17:40 [base.py:148] [LLM Worker 0] Level 2 Sleep: Freed 13.61 GiB. 19.40GiB memory is still in use.
[0;36m(EngineCore_DP0 pid=59255)[0;0m [Stage-1] INFO 02-24 05:17:40 [base.py:188] [LLM Worker 0] ACK emitted for Task 5e836a82-39cf-4e2e-a167-c803b1c8ace0
[Stage-1] INFO 02-24 05:17:40 [omni_stage.py:1323] [Stage-1] Sleep ACKs forwarded to Orchestrator
INFO 02-24 05:17:40 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task 5e836a82-39cf-4e2e-a167-c803b1c8ace0 from stage-1
INFO 02-24 05:17:40 [omni_stage.py:566] [Stage-1] Status transitioned to: SLEEPING
INFO 02-24 05:17:41 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-24 05:17:41 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-24 05:17:41 [async_omni.py:894] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-0] INFO 02-24 05:17:41 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: a07df077-3f65-4b9e-983e-f49e0420de88 (Level: 2)
[Stage-0] INFO 02-24 05:17:41 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task a07df077-3f65-4b9e-983e-f49e0420de88 (Level: 2)
[Stage-0] INFO 02-24 05:17:41 [diffusion_worker.py:228] [Diffusion Worker 0] Handshake Received: Task a07df077-3f65-4b9e-983e-f49e0420de88, Level 2
[Stage-0] INFO 02-24 05:17:41 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-24 05:17:42 [diffusion_worker.py:198] [Diffusion Worker 0] Level 2 Sleep: Freed 18.40 GiB. 1.00GiB memory is still in use.
[Stage-0] INFO 02-24 05:17:42 [diffusion_worker.py:246] [Diffusion Worker 0]: ACK emitted. Freed 18.40 GiB.
[Stage-0] INFO 02-24 05:17:42 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-24 05:17:42 [async_omni.py:641] [AsyncOrchestrator] Intercepted wrapped ACK for task a07df077-3f65-4b9e-983e-f49e0420de88 from stage-0
INFO 02-24 05:17:42 [omni_stage.py:566] [Stage-0] Status transitioned to: SLEEPING




6. test_diffusion_vram_lifecycle_audit 

description: 

Unit Test 6: 端到端显存生命周期全循环审计
完整覆盖“活跃 (Active) -> 深度睡眠 (Deep Sleep) -> 重新唤醒 (Restored)”的闭环流程，验证显存池的动态吞吐能力。
   
result:

#### level 1





#### level 2





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