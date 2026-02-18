


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

1. test_model_integrity_post_wakeup
   
   description: 
   
   result:



2. test_task1_physical_reclamation
   
   description: 
   
   result:




3. test_task2_deterministic_handshake

   description: 
   
   result:




4. test_task3_auto_wakeup_protection
   
   description: 
   
   result:



5. test_error_fallback_and_timeout 
   
   description: 
   
   result:





## AMD

MI300X TP=2



1. test_model_integrity_post_wakeup
   
   description: 
   
   result:



2. test_task1_physical_reclamation

   description: 
   
   result:



3. test_task2_deterministic_handshake

   description: 
   
   result:




4. test_task3_auto_wakeup_protection

   description: 
   
   result:



5. test_error_fallback_and_timeout 

   description: 
   
   result: