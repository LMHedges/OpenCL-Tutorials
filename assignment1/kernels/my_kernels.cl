// Histogram kernel using local memory for 16-bit input with variable bins
kernel void hist_local(global const ushort* A, global int* H, int nr_bins, local int* local_hist) {
    int gid = get_global_id(0);    // Global thread ID
    int lid = get_local_id(0);     // Local thread ID within work-group
    int group_size = get_local_size(0); // Number of threads in work-group

    // Initialize local histogram (each thread clears a portion)
    for (int i = lid; i < nr_bins; i += group_size) {
        local_hist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Ensure all local memory is initialized

    // Calculate bin index for this thread's value
    if (gid < get_global_size(0)) { // Ensure we don't exceed image size
        ushort value = A[gid];
        int bin_index = (value * nr_bins) / 65536; // Scale 16-bit value to nr_bins
        if (bin_index >= nr_bins) bin_index = nr_bins - 1; // Clamp to valid range
        
        // Atomically increment local histogram
        atomic_add(&local_hist[bin_index], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE); // Wait for all threads in group to finish

    // Reduce local histogram to global histogram
    if (lid < nr_bins) {
        if (local_hist[lid] > 0) {
            atomic_add(&H[lid], local_hist[lid]);
        }
    }
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

// Blelloch basic exclusive scan for cumulative histogram with variable bins (padded to power of 2)
kernel void scan_bl(global int* A, const int padded_nr_bins) {
    int id = get_global_id(0);
    if (id >= padded_nr_bins) return; // Guard against out-of-bounds access
    int N = padded_nr_bins;
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