#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
    std::cerr << "Application usage:" << std::endl;
    std::cerr << "  -p : select platform " << std::endl;
    std::cerr << "  -d : select device" << std::endl;
    std::cerr << "  -l : list all platforms and devices" << std::endl;
    std::cerr << "  -f : input image file" << std::endl;
    std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
    int platform_id = 0;
    int device_id = 0;
    string image_filename = "mdr16-gs.pgm"; // Default to 16-bit PGM

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    cimg::exception_mode(0);

    try {
        // Load input image (PGM or PPM, 8-bit or 16-bit)
        CImg<unsigned short> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");

        // Setup OpenCL
        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
        cl::CommandQueue queue(context, context.getInfo<CL_CONTEXT_DEVICES>()[0], CL_QUEUE_PROFILING_ENABLE);

        // Load and build kernel code
        cl::Program::Sources sources;
        AddSources(sources, "kernels/my_kernels.cl");
        cl::Program program(context, sources);
        try {
            program.build();
        }
        catch (const cl::Error& err) {
            std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
            throw err;
        }

        // Compute intensity channel for histogram equalization
        size_t width = image_input.width();
        size_t height = image_input.height();
        size_t channels = image_input.spectrum();
        size_t image_size = width * height; // Size of one channel
        CImg<unsigned short> intensity(image_size, 1, 1, 1, 0);

        if (channels == 1) { // PGM (8-bit or 16-bit)
            intensity = image_input;
        } else if (channels == 3) { // PPM (16-bit RGB)
            cimg_forXY(image_input, x, y) {
                unsigned short r = image_input(x, y, 0, 0);
                unsigned short g = image_input(x, y, 0, 1);
                unsigned short b = image_input(x, y, 0, 2);
                intensity(x + y * width) = (unsigned short)(0.299f * r + 0.587f * g + 0.114f * b); // Luminance
            }
        } else {
            throw std::runtime_error("Unsupported number of channels");
        }

        // Device buffers
        cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_size * sizeof(unsigned short));
        cl::Buffer dev_image_output(context, CL_MEM_WRITE_ONLY, image_size * sizeof(unsigned short));
        cl::Buffer dev_histogram(context, CL_MEM_READ_WRITE, 256 * sizeof(unsigned int));
        cl::Buffer dev_cum_histogram(context, CL_MEM_READ_WRITE, 256 * sizeof(unsigned int));
        cl::Buffer dev_lut(context, CL_MEM_READ_WRITE, 65536 * sizeof(unsigned short)); // 16-bit LUT

        // Timing and metrics per step
        struct StepMetrics {
            double transfer_time = 0;
            double kernel_time = 0;
            double total_time = 0;
            size_t work = 0;
            size_t span = 0;
        };
        StepMetrics step1, step2, step3, step4, step5;
        const int BINS = 256;

        // Step 1: Copy intensity to device and initialize histogram
        cl::Event event1a, event1b;
        queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_size * sizeof(unsigned short), intensity.data(), nullptr, &event1a);
        std::vector<unsigned int> zeros(BINS, 0);
        queue.enqueueWriteBuffer(dev_histogram, CL_TRUE, 0, BINS * sizeof(unsigned int), zeros.data(), nullptr, &event1b);
        event1a.wait();
        event1b.wait();
        step1.transfer_time = (event1a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event1a.getProfilingInfo<CL_PROFILING_COMMAND_START>() +
                               event1b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event1b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
        step1.total_time = step1.transfer_time;
        step1.work = image_size + BINS;
        step1.span = 1;

        // Step 2: Calculate histogram
        cl::Event event2a, event2b;
        cl::Kernel hist_kernel(program, "hist_simple");
        hist_kernel.setArg(0, dev_image_input);
        hist_kernel.setArg(1, dev_histogram);
        hist_kernel.setArg(2, BINS);
        queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(image_size), cl::NullRange, nullptr, &event2a);
        event2a.wait();
        step2.kernel_time = (event2a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event2a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

        std::vector<unsigned int> histogram(BINS);
        queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, BINS * sizeof(unsigned int), histogram.data(), nullptr, &event2b);
        event2b.wait();
        step2.transfer_time = (event2b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event2b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
        step2.total_time = step2.kernel_time + step2.transfer_time;
        step2.work = image_size;
        step2.span = 2;

        CImg<unsigned char> hist_img(256, 200, 1, 1, 0);
        const unsigned char white[] = {255};
        unsigned int max_hist = *std::max_element(histogram.begin(), histogram.end());
        for (int x = 0; x < 256; x++) {
            int height = (int)((histogram[x] / (float)max_hist) * 200);
            hist_img.draw_line(x, 200, x, 200 - height, white);
        }
        CImgDisplay disp_hist(hist_img, "Intensity Histogram");

        // Step 3: Calculate cumulative histogram
        cl::Event event3a, event3b;
        cl::Kernel scan_kernel(program, "scan_bl");
        scan_kernel.setArg(0, dev_histogram);
        queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(BINS), cl::NullRange, nullptr, &event3a);
        event3a.wait();
        step3.kernel_time = (event3a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event3a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

        std::vector<unsigned int> cum_histogram(BINS);
        queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, BINS * sizeof(unsigned int), cum_histogram.data(), nullptr, &event3b);
        event3b.wait();
        step3.transfer_time = (event3b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event3b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
        step3.total_time = step3.kernel_time + step3.transfer_time;
        step3.work = 2 * BINS - 1;
        step3.span = (size_t)log2((double)BINS);

        CImg<unsigned char> cum_hist_img(256, 200, 1, 1, 0);
        unsigned int max_cum_hist = cum_histogram[BINS - 1];
        for (int x = 0; x < 256; x++) {
            int height = (int)((cum_histogram[x] / (float)max_cum_hist) * 200);
            cum_hist_img.draw_line(x, 200, x, 200 - height, white);
        }
        CImgDisplay disp_cum_hist(cum_hist_img, "Cumulative Histogram");

        // Step 4: Normalize LUT
        cl::Event event4a, event4b;
        float scale = 65535.0f / (image_input.width() * image_input.height()); // Scale to 16-bit range
        cl::Kernel normalize_kernel(program, "normalize_lut");
        normalize_kernel.setArg(0, dev_histogram);
        normalize_kernel.setArg(1, dev_lut);
        normalize_kernel.setArg(2, scale);
        queue.enqueueNDRangeKernel(normalize_kernel, cl::NullRange, cl::NDRange(65536), cl::NullRange, nullptr, &event4a); // Full 16-bit range
        event4a.wait();
        step4.kernel_time = (event4a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event4a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

        std::vector<unsigned short> lut(65536);
        queue.enqueueReadBuffer(dev_lut, CL_TRUE, 0, 65536 * sizeof(unsigned short), lut.data(), nullptr, &event4b);
        event4b.wait();
        step4.transfer_time = (event4b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event4b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
        step4.total_time = step4.kernel_time + step4.transfer_time;
        step4.work = 65536;
        step4.span = 1;

        CImg<unsigned char> norm_cum_hist_img(256, 200, 1, 1, 0); // Still display 256 bins for visualization
        for (int x = 0; x < 256; x++) {
            int height = (int)((lut[x * 256] / 65535.0f) * 200); // Sample every 256th value for display
            norm_cum_hist_img.draw_line(x, 200, x, 200 - height, white);
        }
        CImgDisplay disp_norm_cum_hist(norm_cum_hist_img, "Normalized Cumulative Histogram");

        // Step 5: Back projection
        cl::Event event5a, event5b;
        cl::Kernel backproject_kernel(program, "back_project");
        backproject_kernel.setArg(0, dev_image_input);
        backproject_kernel.setArg(1, dev_image_output);
        backproject_kernel.setArg(2, dev_lut);
        queue.enqueueNDRangeKernel(backproject_kernel, cl::NullRange, cl::NDRange(image_size), cl::NullRange, nullptr, &event5a);
        event5a.wait();
        step5.kernel_time = (event5a.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event5a.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;

        std::vector<unsigned short> output_buffer(image_size);
        queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, image_size * sizeof(unsigned short), output_buffer.data(), nullptr, &event5b);
        event5b.wait();
        step5.transfer_time = (event5b.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event5b.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-9;
        step5.total_time = step5.kernel_time + step5.transfer_time;
        step5.work = image_size;
        step5.span = 1;

        // Display output
        CImg<unsigned short> output_image(output_buffer.data(), width, height, 1, 1);
        CImgDisplay disp_output(output_image, "Equalized Image");

        // Print timing and complexity results for each step
        std::cout << "Performance Metrics (seconds) and Complexity:\n";
        std::cout << "Step 1: Input Transfer and Initialization\n";
        std::cout << "  Transfer Time: " << step1.transfer_time << "\n";
        std::cout << "  Kernel Time: " << step1.kernel_time << "\n";
        std::cout << "  Total Time: " << step1.total_time << "\n";
        std::cout << "  Work: " << step1.work << " operations\n";
        std::cout << "  Span: " << step1.span << " steps\n";

        std::cout << "Step 2: Histogram Calculation\n";
        std::cout << "  Transfer Time: " << step2.transfer_time << "\n";
        std::cout << "  Kernel Time: " << step2.kernel_time << "\n";
        std::cout << "  Total Time: " << step2.total_time << "\n";
        std::cout << "  Work: " << step2.work << " operations\n";
        std::cout << "  Span: " << step2.span << " steps\n";

        std::cout << "Step 3: Cumulative Histogram\n";
        std::cout << "  Transfer Time: " << step3.transfer_time << "\n";
        std::cout << "  Kernel Time: " << step3.kernel_time << "\n";
        std::cout << "  Total Time: " << step3.total_time << "\n";
        std::cout << "  Work: " << step3.work << " operations\n";
        std::cout << "  Span: " << step3.span << " steps\n";

        std::cout << "Step 4: Normalize LUT\n";
        std::cout << "  Transfer Time: " << step4.transfer_time << "\n";
        std::cout << "  Kernel Time: " << step4.kernel_time << "\n";
        std::cout << "  Total Time: " << step4.total_time << "\n";
        std::cout << "  Work: " << step4.work << " operations\n";
        std::cout << "  Span: " << step4.span << " steps\n";

        std::cout << "Step 5: Back Projection\n";
        std::cout << "  Transfer Time: " << step5.transfer_time << "\n";
        std::cout << "  Kernel Time: " << step5.kernel_time << "\n";
        std::cout << "  Total Time: " << step5.total_time << "\n";
        std::cout << "  Work: " << step5.work << " operations\n";
        std::cout << "  Span: " << step5.span << " steps\n";

        double overall_total_time = step1.total_time + step2.total_time + step3.total_time + step4.total_time + step5.total_time;
        std::cout << "Overall Total Time: " << overall_total_time << " seconds\n";

        // Wait for all windows to close
        while (!disp_input.is_closed() && !disp_output.is_closed() && 
               !disp_hist.is_closed() && !disp_cum_hist.is_closed() && 
               !disp_norm_cum_hist.is_closed() &&
               !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
            disp_input.wait(1);
            disp_output.wait(1);
            disp_hist.wait(1);
            disp_cum_hist.wait(1);
            disp_norm_cum_hist.wait(1);
        }
    }
    catch (const cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
    }
    catch (CImgException& err) {
        std::cerr << "ERROR: " << err.what() << std::endl;
    }

    return 0;
}
