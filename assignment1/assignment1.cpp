#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"
#include <chrono>

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
    string image_filename = "test.pgm";

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
    }

    cimg::exception_mode(0);

    try {
        // Load input image
        CImg<unsigned char> image_input(image_filename.c_str());
        CImgDisplay disp_input(image_input, "Input Image");

        // Setup OpenCL
        cl::Context context = GetContext(platform_id, device_id);
        std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
        cl::CommandQueue queue(context);

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

        // Device buffers
        size_t image_size = image_input.size();
        const int BINS = 256;
        cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_size);
        cl::Buffer dev_image_output(context, CL_MEM_WRITE_ONLY, image_size);
        cl::Buffer dev_histogram(context, CL_MEM_READ_WRITE, BINS * sizeof(unsigned int));
        cl::Buffer dev_cum_histogram(context, CL_MEM_READ_WRITE, BINS * sizeof(unsigned int));
        cl::Buffer dev_lut(context, CL_MEM_READ_WRITE, BINS * sizeof(unsigned char));

        // Timing variables
        auto total_start = std::chrono::high_resolution_clock::now();
        double transfer_time = 0, kernel_time = 0;

        // Copy input to device
        auto t1 = std::chrono::high_resolution_clock::now();
        queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_size, image_input.data());
        transfer_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();

        // Initialize histogram to zero
        std::vector<unsigned int> zeros(BINS, 0);
        queue.enqueueWriteBuffer(dev_histogram, CL_TRUE, 0, BINS * sizeof(unsigned int), zeros.data());

        // Setup kernels
        cl::Kernel hist_kernel(program, "hist_simple");
        cl::Kernel scan_kernel(program, "scan_bl");
        cl::Kernel normalize_kernel(program, "normalize_lut");
        cl::Kernel backproject_kernel(program, "back_project");

        // Calculate histogram
        hist_kernel.setArg(0, dev_image_input);
        hist_kernel.setArg(1, dev_histogram);
        hist_kernel.setArg(2, BINS);
        t1 = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(hist_kernel, cl::NullRange, cl::NDRange(image_size), cl::NullRange);
        queue.finish();
        kernel_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();

        // Read histogram from device for visualization
        std::vector<unsigned int> histogram(BINS);
        queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, BINS * sizeof(unsigned int), histogram.data());
        CImg<unsigned char> hist_img(256, 200, 1, 1, 0);
        const unsigned char white[] = {255};
        unsigned int max_hist = *std::max_element(histogram.begin(), histogram.end());
        for (int x = 0; x < 256; x++) {
            int height = (int)((histogram[x] / (float)max_hist) * 200);
            hist_img.draw_line(x, 200, x, 200 - height, white);
        }
        CImgDisplay disp_hist(hist_img, "Intensity Histogram (Fig. 1b)");

        // Calculate cumulative histogram
        scan_kernel.setArg(0, dev_histogram);
        t1 = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(BINS), cl::NullRange);
        queue.finish();
        kernel_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();

        // Read cumulative histogram from device for visualization
        std::vector<unsigned int> cum_histogram(BINS);
        queue.enqueueReadBuffer(dev_histogram, CL_TRUE, 0, BINS * sizeof(unsigned int), cum_histogram.data());
        CImg<unsigned char> cum_hist_img(256, 200, 1, 1, 0);
        unsigned int max_cum_hist = cum_histogram[BINS - 1]; // Last value is max
        for (int x = 0; x < 256; x++) {
            int height = (int)((cum_histogram[x] / (float)max_cum_hist) * 200);
            cum_hist_img.draw_line(x, 200, x, 200 - height, white);
        }
        CImgDisplay disp_cum_hist(cum_hist_img, "Cumulative Histogram (Fig. 1c)");

        // Normalize LUT
        float scale = 255.0f / (image_input.width() * image_input.height());
        normalize_kernel.setArg(0, dev_histogram);
        normalize_kernel.setArg(1, dev_lut);
        normalize_kernel.setArg(2, scale);
        t1 = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(normalize_kernel, cl::NullRange, cl::NDRange(BINS), cl::NullRange);
        queue.finish();
        kernel_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();

        // Read LUT from device for visualization
        std::vector<unsigned char> lut(BINS);
        queue.enqueueReadBuffer(dev_lut, CL_TRUE, 0, BINS * sizeof(unsigned char), lut.data());
        CImg<unsigned char> norm_cum_hist_img(256, 200, 1, 1, 0);
        for (int x = 0; x < 256; x++) {
            int height = (int)((lut[x] / 255.0f) * 200);
            norm_cum_hist_img.draw_line(x, 200, x, 200 - height, white);
        }
        CImgDisplay disp_norm_cum_hist(norm_cum_hist_img, "Normalized Cumulative Histogram (Fig. 1d)");

        // Back projection
        backproject_kernel.setArg(0, dev_image_input);
        backproject_kernel.setArg(1, dev_image_output);
        backproject_kernel.setArg(2, dev_lut);
        t1 = std::chrono::high_resolution_clock::now();
        queue.enqueueNDRangeKernel(backproject_kernel, cl::NullRange, cl::NDRange(image_size), cl::NullRange);
        queue.finish();
        kernel_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();

        // Read output
        std::vector<unsigned char> output_buffer(image_size);
        t1 = std::chrono::high_resolution_clock::now();
        queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, image_size, output_buffer.data());
        transfer_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count();

        auto total_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_start).count();

        // Display output
        CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), 1, 1);
        CImgDisplay disp_output(output_image, "Equalized Image");

        // Print timing results
        std::cout << "Performance Metrics (seconds):\n";
        std::cout << "Kernel Execution Time: " << kernel_time << "\n";
        std::cout << "Memory Transfer Time: " << transfer_time << "\n";
        std::cout << "Total Time: " << total_time << "\n";

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