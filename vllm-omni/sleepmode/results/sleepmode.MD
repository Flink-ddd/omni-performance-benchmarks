


## processing

stage 0 for ready,    successful
stage 1 for ready,    successful
stage 2 for ready,    successful

waiting all worker send Handshake Received, successful

Logic for handling diffusion sleep and wakeup memory changes,  successful


## next step

Logic for handling llm and generation sleep and wakeup memory changes, process pending.












## NVIDIA 

RTX 3090 TP=2
RTX A5000 TP=2

1. test_llm_sleep_ack
   
description: 
   
result:



2. test_diffusion_sleep_handshake
   
description: 
   

result:




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



5. test_coordinated_cross_device
   
description: 
   
result:


6. test_auto_wakeup_llm
   
description: 
   
result:


7. test_error_fallback_and_timeout
   
description: 
   
result:




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



6. test_auto_wakeup_llm
   
description: 
   
result:



1. test_error_fallback_and_timeout
   
description: 
   
result:
