#include <iostream>
#include <vector>
#include <string>
#include <cmath> // For std::ceil and std::log2
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

// Function to display command-line usage instructions
void print_help() {
    std::cerr << "Application usage:" << std::endl;
    std::cerr << "  -p : select platform" << std::endl;
    std::cerr << "  -d : select device" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -f : input image file" << std::endl;
    std::cerr << "  -b : number of bins (default 256)" << std::endl;
    std::cerr << "  -s : scan type (bl for Blelloch, hs for Hillis-Steele, default bl)" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

// Helper function to compute the next power of 2 (used for Blelloch scan padding)
size_t next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

int main(int argc, char **argv) {
    // Default parameters
    int platform_id = 0;                // Default OpenCL platform ID
    int device_id = 0;                  // Default OpenCL device ID
    std::string image_filename = "mdr16.ppm"; // Default input image (16-bit RGB PPM)
    int num_bins = 256;                 // Default number of histogram bins
    std::string scan_type = "bl";       // Default scan type (Blelloch)

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
        else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { num_bins = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-s") == 0) && (i < (argc - 1))) { scan_type = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    // Validate number of bins
    if (num_bins <= 0) {
        std::cerr << "Error: Number of bins must be positive" << std::endl;
        return 1;
    }

    // Pad num_bins to the next power of 2 for Blelloch scan compatibility
    size_t padded_num_bins = next_power_of_2(num_bins);

    // Validate scan type
    if (scan_type != "bl" && scan_type != "hs") {
        std::cerr << "Error: Invalid scan type '" << scan_type << "'. Use 'bl' for Blelloch or 'hs' for Hillis-Steele." << std::endl;
        return 1;
    }

    // Disable CImg exceptions for cleaner error handling
    cimg::exception_mode(0);

    try {
        // Load the input image
        CImg<unsigned short> image_input;

        // Determine bit depth from PPM header
        FILE* file = fopen(image_filename.c_str(), "rb");
        if (!file) throw CImgIOException("Cannot open file");
        
        char magic[3] = {0};
        int maxval = 0;
        fscanf(file, "%2s %*d %*d %d", magic, &maxval);
        fclose(file);

        bool is_8bit = (maxval <= 255);

        // Load image based on bit depth
        if (is_8bit) {
            // 8-bit image: Load and scale to 16-bit
            CImg<unsigned char> image_8bit(image_filename.c_str());
            image_input.assign(image_8bit.width(), image_8bit.height(), 1, image_8bit.spectrum());
            cimg_forXYC(image_input, x, y, c) {
                image_input(x, y, 0, c) = (unsigned short)(image_8bit(x, y, 0, c) * 257); // Scale 0-255 to 0-65535
            }
        } else {
            // 16-bit image: Load directly
            image_input = CImg<unsigned short>(image_filename.c_str());
        }

        // Display the input image
        CImgDisplay disp_input(image_input, "Input Image");

        // Set up OpenCL context and queue
        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
        cl::CommandQueue queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

        // Load and build OpenCL kernel code
        cl::Program::Sources sources;
        AddSources(sources, "kernels/my_kernels.cl");
        cl::Program program(context, sources);
        try {
            program.build();
        } catch (const cl::Error& err) {
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        // Extract image properties
        size_t width = image_input.width();
        size_t height = image_input.height();
        size_t channels = image_input.spectrum(); // 1 for grayscale, 3 for RGB
        size_t image_size = width * height;       // Number of pixels in one channel

        // Separate image into channels
        std::vector<CImg<unsigned short>> input_channels(channels);
        for (int c = 0; c < channels; c++) {
            input_channels[c] = CImg<unsigned short>(width, height, 1, 1);
            cimg_forXY(image_input, x, y) {
                input_channels[c](x, y) = image_input(x, y, 0, c);
            }
        }

        // Define device buffers for each channel
        std::vector<cl::Buffer> dev_image_input(channels);
        std::vector<cl::Buffer> dev_image_output(channels);
        std::vector<cl::Buffer> dev_histogram(channels);
        std::vector<cl::Buffer> dev_cum_histogram(channels);
        std::vector<cl::Buffer> dev_lut(channels);

        size_t hist_size = (scan_type == "bl") ? padded_num_bins : num_bins; // Histogram size depends on scan type
        for (int c = 0; c < channels; c++) {
            dev_image_input[c] = cl::Buffer(context, CL_MEM_READ_ONLY, image_size * sizeof(unsigned short));
            dev_image_output[c] = cl::Buffer(context, CL_MEM_WRITE_ONLY, image_size * sizeof(unsigned short));
            dev_histogram[c] = cl::Buffer(context, CL_MEM_READ_WRITE, hist_size * sizeof(unsigned int));
            dev_cum_histogram[c] = cl::Buffer(context, CL_MEM_READ_WRITE, hist_size * sizeof(unsigned int));
            dev_lut[c] = cl::Buffer(context, CL_MEM_READ_WRITE, 65536 * sizeof(unsigned short)); // Full 16-bit LUT
        }

        // Structure to hold timing and complexity metrics for each step
        struct StepMetrics {
            double transfer_time = 0;
            double kernel_time = 0;
            double total_time = 0;
            size_t work = 0; // Total operations
            size_t span = 0; // Critical path length
        };
        std::vector<std::vector<StepMetrics>> metrics(channels, std::vector<StepMetrics>(5));

        // Visualization displays for each channel
        std::vector<CImgDisplay> disp_hist(channels);
        std::vector<CImgDisplay> disp_cum_hist(channels);
        std::vector<CImgDisplay> disp_norm_cum_hist(channels);

        // Process each channel
        for (int c = 0; c < channels; c++) {
            // Step 1: Copy channel data to device and initialize histogram
            cl::Event event1a, event1b;
            queue.enqueueWriteBuffer(dev_image_input[c], CL_TRUE, 0, image_size * sizeof(unsigned short), input_channels[c].data(), nullptr, &event1a);
            std::vector<unsigned int> zeros(hist_size, 0);
            queue.enqueueWriteBuffer(dev_histogram[c], CL_TRUE, 0, hist_size * sizeof(unsigned int), zeros.data(), nullptr, &event1b);
            event1a.wait();
            event1b.wait();
            metrics[c][0].transfer_time = (event1a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event1a.getProfilingInfo<CL_PROFILING_COMMAND_START>() +
                                           event1b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event1b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][0].total_time = metrics[c][0].transfer_time;
            metrics[c][0].work = image_size + hist_size; // Work: Transfer n pixels + init h bins
            metrics[c][0].span = 1;                      // Span: Parallel across infinite processors

            // Step 2: Calculate histogram using local memory
            cl::Event event2a, event2b;
            cl::Kernel hist_kernel(program, "hist_local");
            size_t local_size = 256;                            // Work-group size
            size_t global_size = ((image_size + local_size - 1) / local_size) * local_size; // Round up
            hist_kernel.setArg(0, dev_image_input[c]);
            hist_kernel.setArg(1, dev_histogram[c]);
            hist_kernel.setArg(2, num_bins);
            hist_kernel.setArg(3, cl::Local(num_bins * sizeof(unsigned int)));
            queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(global_size), cl::NDRange(local_size), nullptr, &event2a);
            event2a.wait();
            metrics[c][1].kernel_time = (event2a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event2a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            std::vector<unsigned int> histogram(hist_size);
            queue.enqueueReadBuffer(dev_histogram[c], CL_TRUE, 0, hist_size * sizeof(unsigned int), histogram.data(), nullptr, &event2b);
            event2b.wait();
            metrics[c][1].transfer_time = (event2b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event2b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][1].total_time = metrics[c][1].kernel_time + metrics[c][1].transfer_time;
            metrics[c][1].work = image_size + num_bins;                    // Work: n bin calcs + h reductions
            metrics[c][1].span = (size_t)std::ceil(std::log2((double)image_size / local_size)) + 1; // Span: log(n/L) + sync

            // Visualize histogram
            CImg<unsigned char> hist_img(num_bins, 200, 1, 1, 0);
            const unsigned char white[] = {255};
            unsigned int max_hist = *std::max_element(histogram.begin(), histogram.begin() + num_bins);
            for (int x = 0; x < num_bins; x++) {
                int height = (int)((histogram[x] / (float)max_hist) * 200);
                hist_img.draw_line(x, 200, x, 200 - height, white);
            }
            disp_hist[c] = CImgDisplay(hist_img, ("Histogram Channel " + std::to_string(c + 1)).c_str());

            // Step 3: Calculate cumulative histogram
            cl::Event event3a, event3b;
            if (scan_type == "bl") {
                cl::Kernel scan_kernel(program, "scan_bl");
                scan_kernel.setArg(0, dev_histogram[c]);
                scan_kernel.setArg(1, (int)padded_num_bins);
                queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(padded_num_bins), cl::NDRange(padded_num_bins), nullptr, &event3a);
                event3a.wait();
                metrics[c][2].kernel_time = (event3a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event3a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
                metrics[c][2].work = 2 * padded_num_bins - 1;         // Work: Blelloch O(h)
                metrics[c][2].span = (size_t)std::ceil(std::log2((double)padded_num_bins)); // Span: O(log h)
            } else if (scan_type == "hs") {
                cl::Kernel scan_kernel(program, "scan_hs");
                scan_kernel.setArg(0, dev_histogram[c]);
                scan_kernel.setArg(1, dev_cum_histogram[c]);
                queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(num_bins), cl::NullRange, nullptr, &event3a);
                event3a.wait();
                metrics[c][2].kernel_time = (event3a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event3a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
                metrics[c][2].work = num_bins * (size_t)std::ceil(std::log2((double)num_bins)); // Work: Hillis-Steele O(h log h)
                metrics[c][2].span = (size_t)std::ceil(std::log2((double)num_bins));           // Span: O(log h)
                queue.enqueueCopyBuffer(dev_cum_histogram[c], dev_histogram[c], 0, 0, num_bins * sizeof(unsigned int));
            }
            std::vector<unsigned int> cum_histogram(hist_size);
            queue.enqueueReadBuffer(dev_histogram[c], CL_TRUE, 0, hist_size * sizeof(unsigned int), cum_histogram.data(), nullptr, &event3b);
            event3b.wait();
            metrics[c][2].transfer_time = (event3b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event3b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][2].total_time = metrics[c][2].kernel_time + metrics[c][2].transfer_time;

            // Visualize cumulative histogram
            CImg<unsigned char> cum_hist_img(num_bins, 200, 1, 1, 0);
            unsigned int max_cum_hist = cum_histogram[num_bins - 1];
            for (int x = 0; x < num_bins; x++) {
                int height = (int)((cum_histogram[x] / (float)max_cum_hist) * 200);
                cum_hist_img.draw_line(x, 200, x, 200 - height, white);
            }
            disp_cum_hist[c] = CImgDisplay(cum_hist_img, ("Cumulative Histogram Channel " + std::to_string(c + 1)).c_str());

            // Step 4: Normalize LUT
            cl::Event event4a, event4b;
            float scale = 65535.0f / (image_input.width() * image_input.height()); // Normalization scale
            cl::Kernel normalize_kernel(program, "normalize_lut");
            normalize_kernel.setArg(0, dev_histogram[c]);
            normalize_kernel.setArg(1, dev_lut[c]);
            normalize_kernel.setArg(2, scale);
            normalize_kernel.setArg(3, num_bins);
            queue.enqueueNDRangeKernel(normalize_kernel, cl::NullRange, cl::NDRange(65536), cl::NullRange, nullptr, &event4a);
            event4a.wait();
            metrics[c][3].kernel_time = (event4a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event4a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            std::vector<unsigned short> lut(65536);
            queue.enqueueReadBuffer(dev_lut[c], CL_TRUE, 0, 65536 * sizeof(unsigned short), lut.data(), nullptr, &event4b);
            event4b.wait();
            metrics[c][3].transfer_time = (event4b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event4b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][3].total_time = metrics[c][3].kernel_time + metrics[c][3].transfer_time;
            metrics[c][3].work = 65536; // Work: 65536 operations (one per 16-bit LUT entry)
            metrics[c][3].span = 1;     // Span: O(1) as all entries computed in parallel

            // Visualize normalized cumulative histogram
            CImg<unsigned char> norm_cum_hist_img(num_bins, 200, 1, 1, 0);
            for (int x = 0; x < num_bins; x++) {
                int lut_index = (int)((float)x / num_bins * 65536);
                int height = (int)((lut[lut_index] / 65535.0f) * 200);
                norm_cum_hist_img.draw_line(x, 200, x, 200 - height, white);
            }
            disp_norm_cum_hist[c] = CImgDisplay(norm_cum_hist_img, ("Normalized Cumulative Histogram Channel " + std::to_string(c + 1)).c_str());

            // Step 5: Back projection
            cl::Event event5a, event5b;
            cl::Kernel backproject_kernel(program, "back_project");
            backproject_kernel.setArg(0, dev_image_input[c]);
            backproject_kernel.setArg(1, dev_image_output[c]);
            backproject_kernel.setArg(2, dev_lut[c]);
            queue.enqueueNDRangeKernel(backproject_kernel, cl::NullRange, cl::NDRange(image_size), cl::NullRange, nullptr, &event5a);
            event5a.wait();
            metrics[c][4].kernel_time = (event5a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event5a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            std::vector<unsigned short> output_buffer(image_size);
            queue.enqueueReadBuffer(dev_image_output[c], CL_TRUE, 0, image_size * sizeof(unsigned short), output_buffer.data(), nullptr, &event5b);
            event5b.wait();
            metrics[c][4].transfer_time = (event5b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event5b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
            metrics[c][4].total_time = metrics[c][4].kernel_time + metrics[c][4].transfer_time;
            metrics[c][4].work = image_size; // Work: n lookups
            metrics[c][4].span = 1;          // Span: O(1) parallel

            // Update input channel with equalized data
            cimg_forXY(input_channels[c], x, y) {
                input_channels[c](x, y) = output_buffer[x + y * width];
            }
        }

        // Combine channels into final image
        CImg<unsigned short> output_image(width, height, 1, channels);
        cimg_forXY(output_image, x, y) {
            for (int c = 0; c < channels; c++) {
                output_image(x, y, 0, c) = input_channels[c](x, y);
            }
        }
        CImgDisplay disp_output(output_image, "Equalized Image");

        // Print performance metrics for each channel
        double combined_total_time = 0.0;
        for (int c = 0; c < channels; c++) {
            std::cout << "\nPerformance Metrics (seconds) and Complexity for Channel " << (c + 1)
                      << " (Bins: " << num_bins << (scan_type == "bl" ? ", Padded to " + std::to_string(padded_num_bins) : "")
                      << ", Scan: " << (scan_type == "bl" ? "Blelloch" : "Hillis-Steele") << "):\n";

            double overall_total_time = 0.0;
            for (int step = 0; step < 5; step++) {
                switch (step) {
                    case 0: std::cout << "Step 1: Input Transfer and Initialization\n"; break;
                    case 1: std::cout << "Step 2: Histogram Calculation\n"; break;
                    case 2: std::cout << "Step 3: Cumulative Histogram\n"; break;
                    case 3: std::cout << "Step 4: Normalize LUT\n"; break;
                    case 4: std::cout << "Step 5: Back Projection\n"; break;
                    default: std::cout << "Unknown Step\n"; break;
                }
                std::cout << "  Transfer Time: " << metrics[c][step].transfer_time << "\n";
                std::cout << "  Kernel Time: " << metrics[c][step].kernel_time << "\n";
                std::cout << "  Total Time: " << metrics[c][step].total_time << "\n";
                std::cout << "  Work: " << metrics[c][step].work << " operations\n";
                std::cout << "  Span: " << metrics[c][step].span << " steps\n";
                overall_total_time += metrics[c][step].total_time;
            }
            std::cout << "Overall Total Time for Channel " << (c + 1) << ": " << overall_total_time << " seconds\n";
            combined_total_time += overall_total_time;
        }

        // Print total time for multi-channel images
        if (channels > 1) {
            std::cout << "\nTotal Time for All Channels Combined (RGB Image): " << combined_total_time << " seconds\n";
        }

        // Wait for user to close display windows
        bool all_closed = false;
        while (!all_closed) {
            all_closed = disp_input.is_closed() && disp_output.is_closed();
            for (int c = 0; c < channels; c++) {
                all_closed &= disp_hist[c].is_closed() && disp_cum_hist[c].is_closed() && disp_norm_cum_hist[c].is_closed();
            }
            disp_input.wait(1);
            disp_output.wait(1);
            for (int c = 0; c < channels; c++) {
                disp_hist[c].wait(1);
                disp_cum_hist[c].wait(1);
                disp_norm_cum_hist[c].wait(1);
            }
            if (disp_input.is_keyESC() || disp_output.is_keyESC()) break;
        }
    } catch (const cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
    } catch (CImgException& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
    }

    return 0;
}