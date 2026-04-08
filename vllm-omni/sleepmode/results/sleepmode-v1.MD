

NVIDIA RTX A6000 TP =2 


单元测试1，2，3，4，5，6通过

单元测试5，6  显存变化图已经绘制好













AMD MI300X   tp =2 （3月17号测试）

单元测试1，2，3，4，5，6通过

单元测试5，6  显存变化图已经绘制好




diffusion 的 sleep 流程



INFO 03-24 17:46:58 [async_omni_engine.py:362] [AsyncOmniEngine] Stage 0 engine startup completed
INFO 03-24 17:46:58 [stage_engine_core_client.py:73] [StageEngineCoreClient] Stage-0 initializing EngineCore
INFO 03-24 17:46:58 [stage_engine_core_client.py:104] [StageEngineCoreClient] Stage-0 EngineCore running
INFO 03-24 17:47:08 [async_omni_engine.py:427] [AsyncOmniEngine] Stage 0 initialized
INFO 03-24 17:47:08 [orchestrator.py:158] [Orchestrator] Starting event loop
INFO 03-24 17:47:08 [async_omni_engine.py:288] [AsyncOmniEngine] Orchestrator ready with 2 stages
INFO 03-24 17:47:08 [omni_base.py:105] [AsyncOmni] AsyncOmniEngine initialized in 86.50 seconds
INFO 03-24 17:47:08 [omni_base.py:120] [AsyncOmni] Initialized with 2 stages for model ByteDance-Seed/BAGEL-7B-MoT
INFO 03-24 17:47:08 [async_omni.py:592] [AsyncOrchestrator] Sleep initiated (Task: f60937dd-a2cb-4a92-aaf3-88a798ead5ec). Awaiting 1 ACKs...
(Worker pid=4071111) INFO 03-24 17:47:08 [base.py:236] [LLM Worker 0] Handshake Received: Task f60937dd-a2cb-4a92-aaf3-88a798ead5ec, Level 2
(Worker pid=4071111) INFO 03-24 17:47:14 [cumem.py:216] CuMemAllocator: sleep freed 80.21 GiB memory in total, of which 27.47 GiB is backed up in CPU and the rest 52.74 GiB is discarded directly.
(Worker pid=4071111) INFO 03-24 17:47:14 [base.py:214] [LLM Worker 0] Level 2 Sleep: Freed 80.21 GiB. 30.69GiB memory is still in use.
(Worker pid=4071111) /usr/local/lib/python3.12/dist-packages/torch/distributed/distributed_c10d.py:4876: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
(Worker pid=4071111)   warnings.warn(  # warn only once
(Worker pid=4071111) INFO 03-24 17:47:16 [base.py:269] [LLM Worker 0] ACK emitted for Task f60937dd-a2cb-4a92-aaf3-88a798ead5ec
INFO 03-24 17:47:16 [async_omni.py:89] [Resolver] Task f60937dd-a2cb-4a92-aaf3-88a798ead5ec progress: 1/1 ACKs received.
INFO 03-24 17:47:16 [async_omni.py:96] [Resolver] Task f60937dd-a2cb-4a92-aaf3-88a798ead5ec completed successfully in 8.02s.
INFO 03-24 17:47:16 [async_omni.py:619] [AsyncOrchestrator] Wake-up initiated (Task: 889cb3f4-2d39-4154-b45f-5b7e77ec2f36). Awaiting 1 ACKs...
(Worker pid=4071111) INFO 03-24 17:47:17 [base.py:228] [LLM Worker 0] Wake-up complete.
(Worker pid=4071111) INFO 03-24 17:47:17 [base.py:307] [LLM Worker 0] Wake-up ACK emitted.
INFO 03-24 17:47:17 [async_omni.py:89] [Resolver] Task 889cb3f4-2d39-4154-b45f-5b7e77ec2f36 progress: 1/1 ACKs received.
INFO 03-24 17:47:17 [async_omni.py:96] [Resolver] Task 889cb3f4-2d39-4154-b45f-5b7e77ec2f36 completed successfully in 0.75s.
INFO 03-24 17:47:17 [async_omni.py:632] [AsyncOrchestrator] All 1/1 workers reported WARM for task 889cb3f4-2d39-4154-b45f-5b7e77ec2f36.
INFO 03-24 17:47:17 [omni_base.py:289] [AsyncOmni] Shutting down




单元测试链路：
AsyncOmni -> AsyncOmniEngine -> MQ -> base.py (Worker 进程) -> CuMemAllocator
(证据：日志 (Worker pid=4071111) INFO ... [base.py:214] Level 2 Sleep: Freed 80.21 GiB)

服务端链路：
AsyncOmni -> AsyncOmniEngine -> MQ -> diffusion_worker.py (Worker 子类进程) -> CuMemAllocator
(证据：服务端日志 INFO ... [diffusion_worker.py:301] [Worker 0] Sleep Level 2: physically freed 27.61 GiB)


继续测试



3月17号提交PR，最晚3月18号提交PR。（3月19号提交）

已增加doc

解决review问题，推进合并，推进omni team尽量在25号合并。

目前没有最新的review状态了，还在等待。


目前AMD-ci检查报错



配合omni team的reviewer进行代码修复工作


4月1号
目前已经跑nightly-test测试


4月2-3号
推进合并


目前修复相关的问题


等待合并


进入到组内测试，正在修改脚本。



