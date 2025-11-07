import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch.nn.utils.prune as prune
import torch.quantization as tq

start_load = time.time()

sam_checkpoint = "sam_vit_b_01ec64.pth"  # path to model weights
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.eval()
sam.cpu()

end_load = time.time()
print(f"✅ Model loaded in {(end_load - start_load):.2f} seconds")

print("🔧 Applying pruning...")
for name, module in sam.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.3)

for name, module in sam.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, "weight")

print("Applying dynamic quantization...")
sam_q = tq.quantize_dynamic(
    sam, {torch.nn.Linear}, dtype=torch.qint8
)

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_q.to(device)
print(f"💻 Running on {device.upper()}")

image_path = "fuzzy.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask_generator = SamAutomaticMaskGenerator(sam_q)

print("⏱️ Generating masks...")
start_infer = time.time()
masks = mask_generator.generate(image)
end_infer = time.time()

print(f"✅ Generated {len(masks)} masks in {(end_infer - start_infer):.2f} seconds!")

m = masks[0]['segmentation']
plt.imshow(image)
plt.imshow(m, alpha=0.5)
plt.axis('off')
plt.show()

plt.imshow(image)
plt.imshow(m, alpha=0.5)
plt.axis('off')
plt.savefig("fuzzy_mask.png", bbox_inches='tight', pad_inches=0)
plt.close()

mask = masks[0]['segmentation']
mask_uint8 = (mask * 255).astype(np.uint8)
isolated = cv2.bitwise_and(image, image, mask=mask_uint8)

b, g, r = cv2.split(isolated)
rgba = cv2.merge((b, g, r, mask_uint8))
#cv2.imwrite("isolated_object.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

print("✅ Saved isolated object as 'isolated_object.png'")
