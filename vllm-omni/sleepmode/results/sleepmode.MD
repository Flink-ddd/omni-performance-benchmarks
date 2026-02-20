


## processing

stage 0 for ready,    successful
stage 1 for ready,    successful
stage 2 for ready,    successful

waiting all worker send Handshake Received, successful

Logic for handling diffusion sleep and wakeup memory changes,  successful

Logic for handling llm and generation sleep and wakeup memory changes, successful

test pytest 5, successful


## next step

1. add AMD logic and test AMD







overall


测试项,物理含义,状态
测试 1,LLM 睡眠信号与 10G+ 显存回收,SUCCESS
测试 2-4,Diffusion 睡眠、唤醒及图像一致性,SUCCESS
测试 5,GPU 1 异构引擎协同回收 (Talker + Flux),SUCCESS












## NVIDIA 

RTX A5000 TP=2

1. test_llm_sleep_ack
   
description: 
   
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
   
result:

--- Running test: test_coordinated_cross_device
INFO 02-19 13:50:48 [omni_stage.py:566] [Stage-1] Status transitioned to: TRANSITIONING
INFO 02-19 13:50:48 [omni_stage.py:570] [Stage-1] Submitting SLEEP task (Level: 2)
INFO 02-19 13:50:48 [async_omni.py:885] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
INFO 02-19 13:50:48 [omni_stage.py:566] [Stage-0] Status transitioned to: TRANSITIONING
INFO 02-19 13:50:48 [omni_stage.py:570] [Stage-0] Submitting SLEEP task (Level: 2)
INFO 02-19 13:50:48 [async_omni.py:885] [AsyncOrchestrator] Sleep initiated. Awaiting confirmation from 1 workers...
[Stage-1] INFO 02-19 13:50:48 [async_omni_llm.py:233] [AsyncOmniLLM] Engine Relay: Sleep Task c837f52b-8992-427b-87a0-be67c5ad1de1 (Level 2)
[Stage-0] INFO 02-19 13:50:48 [async_omni_diffusion.py:309] [Entrypoint] Relaying Sleep Task: 6c6e67c4-1be6-495d-95b6-13eb19178c9c (Level: 2)
[Stage-0] INFO 02-19 13:50:48 [diffusion_engine.py:401] [Diffusion Engine Relay] Dispatching Sleep Task 6c6e67c4-1be6-495d-95b6-13eb19178c9c (Level: 2)
[Stage-0] INFO 02-19 13:50:48 [diffusion_worker.py:253] [Diffusion Worker 0] Handshake Received: Task 6c6e67c4-1be6-495d-95b6-13eb19178c9c, Level 2
[0;36m(EngineCore_DP0 pid=144301)[0;0m [Stage-1] INFO 02-19 13:50:48 [base.py:143] [Omni Worker 0] Handshake Received: Task c837f52b-8992-427b-87a0-be67c5ad1de1, Level 2
[0;36m(EngineCore_DP0 pid=144301)[0;0m [Stage-1] INFO 02-19 13:50:48 [cumem.py:213] CuMemAllocator: sleep freed 0.47 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 0.47 GiB is discarded directly.
[0;36m(EngineCore_DP0 pid=144301)[0;0m [Stage-1] INFO 02-19 13:50:48 [base.py:125] [LLM Worker 0] Sleep mode freed 0.57 GiB.
[0;36m(EngineCore_DP0 pid=144301)[0;0m [Stage-1] INFO 02-19 13:50:48 [base.py:168] [Omni Worker 0] ACK emitted for Task c837f52b-8992-427b-87a0-be67c5ad1de1
[Stage-1] INFO 02-19 13:50:48 [omni_stage.py:1323] [Stage-1] Sleep ACKs forwarded to Orchestrator
INFO 02-19 13:50:48 [async_omni.py:632] [AsyncOrchestrator] Intercepted wrapped ACK for task c837f52b-8992-427b-87a0-be67c5ad1de1 from stage-1
[Stage-0] INFO 02-19 13:50:48 [cumem.py:213] CuMemAllocator: sleep freed 14.89 GiB memory in total, of which 0.00 GiB is backed up in CPU and the rest 14.89 GiB is discarded directly.
[Stage-0] INFO 02-19 13:50:49 [diffusion_worker.py:209] Sleep mode freed 0.00 GiB memory, 14.89 GiB memory is still in use.
[Stage-0] INFO 02-19 13:50:49 [diffusion_worker.py:275] [Diffusion Worker 0]: ACK emitted. Freed 14.87 GiB.
[Stage-0] INFO 02-19 13:50:49 [omni_stage.py:1323] [Stage-0] Sleep ACKs forwarded to Orchestrator
INFO 02-19 13:50:49 [async_omni.py:632] [AsyncOrchestrator] Intercepted wrapped ACK for task 6c6e67c4-1be6-495d-95b6-13eb19178c9c from stage-0










## AMD

MI300X TP=2



1. test_llm_sleep_ack
   
description: 
   
result:



2. test_diffusion_sleep_handshake
   
description: 
   
result:




3. test_cross_device_cleanup

description: 
   
result:




4. test_diffusion_integrity_bit_level
   
description: 
   
result:




5. test_coordinated_cross_device
   
description: 
   
result:

