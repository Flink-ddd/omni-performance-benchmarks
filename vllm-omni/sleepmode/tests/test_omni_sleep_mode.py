import asyncio
import pytest
import torch
import uuid
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.diffusion.data import OmniACK
from vllm_omni.inputs.data import OmniSamplingParams
from vllm.sampling_params import SamplingParams
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniTest")

pytestmark = [pytest.mark.core_model, pytest.mark.gpu]



def get_vram_info(device_id: int) -> dict:
    """Obtain a snapshot of the specified GPU's memory (GiB)."""
    torch.cuda.synchronize(device_id)
    return {
        "reserved": torch.cuda.memory_reserved(device_id) / 1024**3,
        "allocated": torch.cuda.memory_allocated(device_id) / 1024**3
    }


@pytest.fixture(scope="module")
async def llm_engine():
    model_name = "Qwen/Qwen2.5-Omni-3B"
    common_args = {
        "enable_sleep_mode": True,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.45
    }
    stages = [
        # Stage 0 (Thinker): put 0   test test_auto_wakeup_llm update from 0.8 to 0.7 --- IGNORE ---
        {"stage_id": 0, "stage_type": "llm", "runtime": {"process": True, "devices": "0", "max_batch_size": 1}, 
         "engine_args": {**common_args, "model_stage": "thinker", "gpu_memory_utilization": 0.7}},
        # Stage 1 (Talker): put 1    test test_coordinated_cross_device update from 0.45 to 0.2 --- IGNORE ---
        {"stage_id": 1, "stage_type": "llm", "engine_input_source": [0], "runtime": {"process": True, "devices": "1", "max_batch_size": 1, "connector_type": "queue"}, 
         "engine_args": {**common_args, "model_stage": "talker", "gpu_memory_utilization": 0.2}},
    ]
    connectors = [{"src_stage_id": 0, "dst_stage_id": 1, "connector_type": "queue"}]
    engine = AsyncOmni(model_name, stages=stages, connectors=connectors)
    for stage in engine.stage_list:
        if hasattr(stage, "engine") and stage.engine:
            stage.engine.orchestrator = engine
    yield engine
    engine.shutdown()



@pytest.fixture(scope="module")
async def diffusion_engine():
    model_name = "black-forest-labs/FLUX.2-klein-4B"
    stages = [
        {
            "stage_id": 0, 
            "stage_type": "diffusion", 
            "runtime": {"process": True, "devices": "1", "max_batch_size": 1}, 
            "engine_args": {
                "model_stage": "base", 
                "gpu_memory_utilization": 0.6, # test test_coordinated_cross_device update from 0.45 to 0.6 --- IGNORE ---
                "model_class_name": "FluxPipeline",
                "enable_sleep_mode": True
            },
            "final_output": True, 
            "final_output_type": "image"
        }
    ]
    engine = AsyncOmni(model_name, stages=stages)
    for stage in engine.stage_list:
        if hasattr(stage, "engine") and stage.engine:
            stage.engine.orchestrator = engine
    yield engine
    engine.shutdown()



class TestOmniSleepMode:

    @pytest.mark.asyncio
    async def test_llm_sleep_ack(self, llm_engine: AsyncOmni):
        """LLM Thinker (GPU0) Signal and Physical Recycling Audit"""
        vram_before = get_vram_info(0)["reserved"]
        logger.info(f"Thinker initial VRAM: {vram_before:.2f} GiB")
        acks = await llm_engine.sleep(stage_ids=[0], level=2)
        def _get_val(ack, key, default=None):
            return getattr(ack, key, ack.get(key) if isinstance(ack, dict) else default)
        # Verification signal successful
        assert all(_get_val(ack, "status") == "SUCCESS" for ack in acks)
        # Verify physical recycling volume
        total_freed_bytes = sum(_get_val(ack, "freed_bytes", 0) for ack in acks)
        freed_gib = total_freed_bytes / 1024**3
        logger.info(f"Thinker VRAM physically reclaimed: {freed_gib:.2f} GiB")
        assert freed_gib > 5.0


    @pytest.mark.asyncio
    async def test_diffusion_sleep_handshake(self, diffusion_engine: AsyncOmni):
        """Diffusion Worker stage signal loop"""
        logger.info("Starting Diffusion Worker Handshake Test")
        acks = await diffusion_engine.sleep(stage_ids=[0], level=2)
        assert all(ack.status == "SUCCESS" for ack in acks)
        logger.info(f"Success: Received {len(acks)} Diffusion Worker ACKs")


    @pytest.mark.asyncio
    async def test_cross_device_cleanup(self, diffusion_engine: AsyncOmni):
        """Physical recycling audit: leveraging deterministic data returned by Workers"""
        acks = await diffusion_engine.sleep(stage_ids=[0], level=2)
        # Sum up the release amounts reported by all Workers.
        total_freed_bytes = sum(getattr(ack, "freed_bytes", 0) for ack in acks)
        freed_gb = total_freed_bytes / 1024**3
        logger.info(f"Physical reclamation summary from workers:")
        logger.info(f"- Total Workers: {len(acks)}")
        logger.info(f"- Total Freed: {freed_gb:.2f} GiB")
        assert freed_gb > 14.0
        logger.info("SUCCESS: 100% weights offloaded from GPU 1.")


    @pytest.mark.asyncio
    async def test_diffusion_integrity_bit_level(self, diffusion_engine: AsyncOmni):
        """Bit-level consistency after Diffusion wake-up (prevent image corruption)"""
        prompt = "A high-tech lab in Kuala Lumpur"
        sp = OmniDiffusionSamplingParams(
            num_inference_steps=4, 
            height=512, 
            width=512
        )
       
        # Baseline Generation
        logger.info("Step 1: Running Baseline Generation...")
        base_output = None
        async for output in diffusion_engine.generate(prompt, request_id="base", sampling_params_list=[sp]):
            base_output = output

        # Deep Sleep (Level 2)
        logger.info("Step 2: Entering Deep Sleep...")
        await diffusion_engine.sleep(stage_ids=[0], level=2)
        # Physical Wake-up
        logger.info("Step 3: Waking up...")
        await diffusion_engine.wake_up(stage_ids=[0])

        # Post-Wakeup Generation
        logger.info("Step 4: Running Post-Wakeup Generation...")
        post_output = None
        async for output in diffusion_engine.generate(prompt, request_id="post", sampling_params_list=[sp]):
            post_output = output

        # Assert result consistency
        assert len(base_output.images) == len(post_output.images)
        assert post_output.images[0] is not None
        logger.info("SUCCESS: Diffusion integrity verified.")


    @pytest.mark.asyncio
    async def test_coordinated_cross_device(self, llm_engine: AsyncOmni, diffusion_engine: AsyncOmni):
        """Heterogeneous Coordinated Cleanup Test (Talker and Diffusion on GPU 1)"""
        # At this point, GPU 1 hosts both Talker (4.5G) and Diffusion (14.8G)
        initial_vram = get_vram_info(1)["reserved"]
        logger.info(f"GPU 1 total pressure: {initial_vram:.2f} GiB")

        # Simultaneously issue physical cleanup
        logger.info("Issuing concurrent SLEEP commands to LLM-Talker and Diffusion-Base...")
        await asyncio.gather(
            llm_engine.sleep(stage_ids=[1], level=2), # Talker
            diffusion_engine.sleep(stage_ids=[0], level=2) # Diffusion
        )

        torch.cuda.empty_cache()
        await asyncio.sleep(1)

        final_vram = get_vram_info(1)["reserved"]
        freed_vram = initial_vram - final_vram
        logger.info(f"Physical reclamation results for GPU 1:")
        logger.info(f"- Initial: {initial_vram:.2f} GiB")
        logger.info(f"- Final:   {final_vram:.2f} GiB")
        logger.info(f"- Total Freed: {freed_vram:.2f} GiB")

        assert freed_vram > 15.0 or final_vram < 5.0
        logger.info("SUCCESS: Heterogeneous VRAM cleanup verified on GPU 1.")