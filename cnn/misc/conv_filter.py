#!/usr/bin/env python3

# ((width_or_height - kernel_size + 2 * padding) / stride) + 1
image_size = 256
kernel_size = 4
padding = 1
stride = 2

calculate_image_size = (
    lambda: ((image_size - kernel_size + (2 * padding)) / stride) + 1
)

print(
    f"Input image size: {image_size} | Output image size: {calculate_image_size()}"
)
