__kernel void gaussian_blur_rgba(
    __global const uchar* input,
    __global uchar* output,
    const int width,
    const int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= width || y >= height)
        return;

    // Define kernel inline (Metal-compatible: no const/global static arrays)
    float r = 0.0f, g = 0.0f, b = 0.0f;
    int idx;
    
    int kx[3] = {-1, 0, 1};
    int ky[3] = {-1, 0, 1};
    float kernel_weights[9] = {
        1.0f/16, 2.0f/16, 1.0f/16,
        2.0f/16, 4.0f/16, 2.0f/16,
        1.0f/16, 2.0f/16, 1.0f/16
    };

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int nx = clamp(x + kx[j], 0, width - 1);
            int ny = clamp(y + ky[i], 0, height - 1);
            idx = (ny * width + nx) * 4;
            float weight = kernel_weights[i * 3 + j];
            r += input[idx + 0] * weight;
            g += input[idx + 1] * weight;
            b += input[idx + 2] * weight;
        }
    }

    int out_idx = (y * width + x) * 4;
    output[out_idx + 0] = (uchar)clamp((int)(r + 0.5f), 0, 255);
    output[out_idx + 1] = (uchar)clamp((int)(g + 0.5f), 0, 255);
    output[out_idx + 2] = (uchar)clamp((int)(b + 0.5f), 0, 255);
    output[out_idx + 3] = input[out_idx + 3]; // Preserve alpha
}


// Kernel 2: Logarithmic Tone Mapping
__kernel void log_tone_map_rgba(
    __global const uchar* input,
    __global uchar* output,
    const int width,
    const int height,
    const float max_luminance
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int idx = (y * width + x) * 4;

    // Convert to float and normalize RGB values to [0, 1]
    float r = input[idx + 0] / 255.0f;
    float g = input[idx + 1] / 255.0f;
    float b = input[idx + 2] / 255.0f;

    // Compute luminance Y using Rec. 709 formula
    float Y = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    // Apply logarithmic tone mapping
    float Y_out = log(1.0f + Y) / log(1.0f + max_luminance);

    // Avoid division by zero (when Y == 0)
    float scale = (Y > 0.0f) ? (Y_out / Y) : 0.0f;

    // Apply tone-mapped luminance to RGB while preserving ratio
    float r_out = clamp(r * scale, 0.0f, 1.0f);
    float g_out = clamp(g * scale, 0.0f, 1.0f);
    float b_out = clamp(b * scale, 0.0f, 1.0f);

    // Convert back to 8-bit unsigned integers
    output[idx + 0] = (uchar)(r_out * 255.0f + 0.5f);
    output[idx + 1] = (uchar)(g_out * 255.0f + 0.5f);
    output[idx + 2] = (uchar)(b_out * 255.0f + 0.5f);
    output[idx + 3] = input[idx + 3];  // Preserve Alpha
}
