import os
from IPython.display import Image
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from sam3.visualization_utils import plot_results
import asyncio

# Setup SAM3
SAM3_ROOT = os.path.dirname(r"C:\Users\amogh\Desktop\sam3")
os.chdir(SAM3_ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_ = os.system("nvidia-smi")

sam3_root = os.path.dirname(sam3.__file__)
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
# We initialize processor once, but note that set_image modifies state.
# Since we are creating a new processor instance inside the function in the original code,
# we should keep that pattern to avoid race conditions if multiple requests came in (though model inference is likely single-threaded on GPU).
# However, to avoid reloading the model (which is heavy), 'model' is global. 'Sam3Processor' is lightweight.


async def sam3_api(image_path, prompt):
    loop = asyncio.get_running_loop()

    # 1. Load Image (Fast IO)
    image = Image.open(image_path)
    image = image.convert("RGB")

    # 2. Initialize Processor (Fast)
    processor = Sam3Processor(model, confidence_threshold=0.5)

    # 3. Offload Heavy Blocking Tasks to Thread
    # set_image and set_text_prompt are CPU/GPU bound and blocking.
    # We run them in the default executor (thread pool) to keep the async loop alive.

    # Using a partial or lambda to pass arguments
    def run_inference():
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state, prompt=prompt
        )
        return inference_state

    # This 'await' yields control back to the event loop while the thread works.
    inference_state = await loop.run_in_executor(None, run_inference)

    # 4. Plotting (matplotlib is also blocking/slow, so good to offload)
    def create_plot():
        plot_results(image, inference_state)
        fig = plt.gcf()
        plt.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)
        return result_image

    result_image = await loop.run_in_executor(None, create_plot)

    return result_image, inference_state["scores"]
