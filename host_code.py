import pyopencl as cl
import numpy as np
from PIL import Image

# 1️⃣ Load RGBA Image from Disk
img = Image.open("Input_images/optimus-prime.png").convert("RGBA")
img_np = np.array(img, dtype=np.uint8)
height, width, channels = img_np.shape
img_flat = img_np.flatten()

# 2️⃣ OpenCL Initialization (Platform, Device, Context, Queue)
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# 3️⃣ Load and Compile OpenCL Kernels from .cl File
with open("kernels.cl", "r") as f:
    kernel_source = f.read()
program = cl.Program(ctx, kernel_source).build()

# 4️⃣ Allocate Buffers (input, intermediate, output)
input_buf       = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=img_flat)
blurred_buf     = cl.Buffer(ctx, mf.READ_WRITE, img_flat.nbytes)  # Intermediate
final_output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, img_flat.nbytes)  # Final output

# 5️⃣ Set Arguments and Enqueue Gaussian Blur Kernel (Kernel 1)
kernel1 = program.gaussian_blur_rgba
kernel1.set_args(input_buf, blurred_buf, np.int32(width), np.int32(height))
event1 = cl.enqueue_nd_range_kernel(queue, kernel1, (width, height), None)

# 6️⃣ Ensure Kernel 1 Completes Before Kernel 2 Starts
event1.wait()

# 7️⃣ Set Arguments and Enqueue Tone Mapping Kernel (Kernel 2)
max_luminance = np.float32(5.0)
kernel2 = program.log_tone_map_rgba
kernel2.set_args(blurred_buf, final_output_buf, np.int32(width), np.int32(height), max_luminance)
event2 = cl.enqueue_nd_range_kernel(queue, kernel2, (width, height), None)

# 8️⃣ Wait for Completion of Kernel 2
event2.wait()
queue.finish()

# 9️⃣ Read Final Processed Image from GPU to CPU
result_flat = np.empty_like(img_flat)
cl.enqueue_copy(queue, result_flat, final_output_buf).wait()
result_img = result_flat.reshape((height, width, 4))

# 🔟 Save the Final Image
Image.fromarray(result_img, mode="RGBA").save("Output_images/final_optimus.png")
print("Saved as final_optimus.png")
