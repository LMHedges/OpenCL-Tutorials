// A simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
    int id = get_global_id(0);
    B[id] = A[id];
}

// Filter to extract red channel (example, unchanged)
kernel void filter_r(global const uchar* A, global uchar* B) {
    int id = get_global_id(0);
    int image_size = get_global_size(0) / 3; // Each image consists of 3 colour channels
    int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

    if (colour_channel == 0) {
        B[id] = A[id];
    } else {
        B[id] = 0;
    }
}

// Simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
    int width = get_global_size(0); // Image width in pixels
    int height = get_global_size(1); // Image height in pixels
    int image_size = width * height; // Image size in pixels
    int channels = get_global_size(2); // Number of colour channels: 3 for RGB

    int x = get_global_id(0); // Current x coord
    int y = get_global_id(1); // Current y coord
    int c = get_global_id(2); // Current colour channel

    int id = x + y * width + c * image_size; // Global id in 1D space

    B[id] = A[id];
}

// 2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
    int width = get_global_size(0); // Image width in pixels
    int height = get_global_size(1); // Image height in pixels
    int image_size = width * height; // Image size in pixels
    int channels = get_global_size(2); // Number of colour channels: 3 for RGB

    int x = get_global_id(0); // Current x coord
    int y = get_global_id(1); // Current y coord
    int c = get_global_id(2); // Current colour channel

    int id = x + y * width + c * image_size; // Global id in 1D space

    uint result = 0;

    for (int i = max(x - 1, 0); i <= min(x + 1, width - 1); i++)
        for (int j = max(y - 1, 0); j <= min(y + 1, height - 1); j++)
            result += A[i + j * width + c * image_size];

    result /= 9;

    B[id] = (uchar)result;
}

// 2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
    int width = get_global_size(0); // Image width in pixels
    int height = get_global_size(1); // Image height in pixels
    int image_size = width * height; // Image size in pixels
    int channels = get_global_size(2); // Number of colour channels: 3 for RGB

    int x = get_global_id(0); // Current x coord
    int y = get_global_id(1); // Current y coord
    int c = get_global_id(2); // Current colour channel

    int id = x + y * width + c * image_size; // Global id in 1D space

    float result = 0;

    for (int i = max(x - 1, 0); i <= min(x + 1, width - 1); i++)
        for (int j = max(y - 1, 0); j <= min(y + 1, height - 1); j++)
            result += A[i + j * width + c * image_size] * mask[(i - (x - 1)) + (j - (y - 1)) * 3];

    B[id] = (uchar)result;
}

// RGB to grayscale (for 8-bit RGB)
kernel void rgb2grey(global const uchar* A, global uchar* B) {
    int id = get_global_id(0);
    int image_size = get_global_size(0) / 3; // Each image consists of 3 colour channels
    int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

    if (colour_channel == 0) {
        B[id] = 255 - ((A[id] * 0.2126) + (A[id + image_size] * 0.7152) + (A[id + (image_size * 2)] * 0.0722));
    } else if (colour_channel == 1) {
        B[id] = 255 - ((A[id - image_size] * 0.2126) + (A[id] * 0.7152) + (A[id + image_size] * 0.0722));
    } else {
        B[id] = 255 - ((A[id - (image_size * 2)] * 0.2126) + (A[id - image_size] * 0.7152) + (A[id] * 0.0722));
    }
}

// Fixed 4-step reduce
kernel void reduce_add_1(global const int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);

    B[id] = A[id]; // Copy input to output

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (((id % 2) == 0) && ((id + 1) < N))
        B[id] += B[id + 1];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (((id % 4) == 0) && ((id + 2) < N))
        B[id] += B[id + 2];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (((id % 8) == 0) && ((id + 4) < N))
        B[id] += B[id + 4];

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (((id % 16) == 0) && ((id + 8) < N))
        B[id] += B[id + 8];
}

// Flexible step reduce
kernel void reduce_add_2(global const int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);

    B[id] = A[id];

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (!(id % (i * 2)) && ((id + i) < N))
            B[id] += B[id + i];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// Reduce using local memory
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    scratch[lid] = A[id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (!(lid % (i * 2)) && ((lid + i) < N))
            scratch[lid] += scratch[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    B[id] = scratch[lid];
}

// Reduce using local memory + accumulation
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    scratch[lid] = A[id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (!(lid % (i * 2)) && ((lid + i) < N))
            scratch[lid] += scratch[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!lid) {
        atomic_add(&B[0], scratch[lid]);
    }
}

// Histogram kernel for 16-bit input (single channel) with variable bins
kernel void hist_simple(global const ushort* A, global int* H, int nr_bins) {
    int id = get_global_id(0);
    ushort value = A[id];
    int bin_index = (value * nr_bins) / 65536; // Scale 16-bit value to nr_bins
    if (bin_index >= nr_bins) bin_index = nr_bins - 1; // Clamp to valid range
    atomic_inc(&H[bin_index]);
}

// Hillis-Steele basic inclusive scan
kernel void scan_hs(global int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);
    global int* C;

    for (int stride = 1; stride < N; stride *= 2) {
        B[id] = A[id];
        if (id >= stride)
            B[id] += A[id - stride];

        barrier(CLK_GLOBAL_MEM_FENCE);

        C = A; A = B; B = C; // Swap A & B
    }
}

// Double-buffered Hillis-Steele scan
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);
    local int *scratch_3;

    scratch_1[lid] = A[id];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid >= i)
            scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
        else
            scratch_2[lid] = scratch_1[lid];

        barrier(CLK_LOCAL_MEM_FENCE);

        scratch_3 = scratch_2;
        scratch_2 = scratch_1;
        scratch_1 = scratch_3;
    }

    B[id] = scratch_1[lid];
}

// Blelloch basic exclusive scan for cumulative histogram with variable bins
kernel void scan_bl(global int* A, const int nr_bins) {
    int id = get_global_id(0);
    if (id >= nr_bins) return; // Guard against out-of-bounds access
    int N = nr_bins;
    int t;

    // Up-sweep
    for (int stride = 1; stride < N; stride *= 2) {
        if (((id + 1) % (stride * 2)) == 0)
            A[id] += A[id - stride];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Down-sweep
    if (id == 0)
        A[N - 1] = 0; // Exclusive scan

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (int stride = N / 2; stride > 0; stride /= 2) {
        if (((id + 1) % (stride * 2)) == 0) {
            t = A[id];
            A[id] += A[id - stride];
            A[id - stride] = t;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// Calculates block sums
kernel void block_sum(global const int* A, global int* B, int local_size) {
    int id = get_global_id(0);
    B[id] = A[(id + 1) * local_size - 1];
}

// Simple exclusive serial scan based on atomic operations
kernel void scan_add_atomic(global int* A, global int* B) {
    int id = get_global_id(0);
    int N = get_global_size(0);
    for (int i = id + 1; i < N && id < N; i++)
        atomic_add(&B[i], A[id]);
}

// Adjust partial scans by adding block sums
kernel void scan_add_adjust(global int* A, global const int* B) {
    int id = get_global_id(0);
    int gid = get_group_id(0);
    A[id] += B[gid];
}

// Normalize LUT kernel for 16-bit output with variable bins
kernel void normalize_lut(global const int* cum_histogram, global ushort* lut, float scale, const int nr_bins) {
    int id = get_global_id(0);
    if (id >= 65536) return; // Guard against out-of-bounds access
    int bin = (id * nr_bins) / 65536; // Map 16-bit value to nr_bins
    if (bin >= nr_bins) bin = nr_bins - 1; // Clamp to valid range
    lut[id] = (ushort)(cum_histogram[bin] * scale); // Scale to 16-bit range
}

// Back projection kernel for 16-bit data
kernel void back_project(global const ushort* input, global ushort* output, global ushort* lut) {
    int id = get_global_id(0);
    output[id] = lut[input[id]];
}
