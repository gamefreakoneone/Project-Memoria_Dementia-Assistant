import os
from functools import partial
from IPython.display import display, Image
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from sam3.visualization_utils import plot_results
import asyncio

SAM3_ROOT = os.path.dirname(r"C:\Users\amogh\Desktop\sam3")
os.chdir(SAM3_ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_ = os.system("nvidia-smi")



sam3_root = os.path.dirname(sam3.__file__)
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)






# image = os.path.abspath(image)
# send_generate_request = partial(send_generate_request_orig, server_url=LLM_SERVER_URL, model=llm_config["model"], api_key=llm_config["api_key"])
# call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
# output_image_path = run_single_image_inference(
#     image, prompt, llm_config, send_generate_request, call_sam_service,
#     debug=True, output_dir="agent_output"
# )

# # display output
# if output_image_path is not None:
#     display(Image(filename=output_image_path))

# async def sam3_api(image_path , prompt):
#     image = Image.open(image_path)
#     image = image.convert("RGB")
#     width, height = image.size
#     processor = Sam3Processor(model, confidence_threshold=0.5)
#     inference_state = processor.set_image(image)
#     processor.reset_all_prompts(inference_state)
#     inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

#     img0 = Image.open(image_path)
#     # plot_results(img0)
#     return img0




async def sam3_api(image_path, prompt):
    image = Image.open(image_path)
    image = image.convert("RGB")
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    plot_results(image, inference_state)
    
    fig = plt.gcf()
    plt.axis('off')
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close(fig)
    return result_image , inference_state["scores"] # This returns the image and the number of objects identified

async def test():
    result = await sam3_api(r"C:\Users\amogh\Pictures\Camera Roll\WIN_20260105_14_22_21_Pro.jpg", "earphones")
    result.save(r"C:\Users\amogh\Desktop\Blue-Dream\Storage\screenshots\test_output.png")
    result.show()  # Opens in default image viewer

asyncio.run(test())