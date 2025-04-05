#include <iostream>
#include <vector>

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 3 - memory allocation
		//host - input
		std::vector<int> A = { 35, 23, 74, 43, 36, 75, 34, 64, 35, 85 }; //C++11 allows this type of initialisation
		std::vector<float> B = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

		// std::vector<int> A(100000000);
		// std::vector<int> B(100000000);
		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size() * sizeof(int);//size in bytes
		size_t vector_size_float = B.size() * sizeof(float);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size_float);

		//Part 4 - device operations

		cl::Event a_event;
		cl::Event b_event;
		cl::Event c_event;

		//4.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0], NULL, &a_event);

		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_avg = cl::Kernel(program, "avg_filter");
		kernel_avg.setArg(0, buffer_A);
		kernel_avg.setArg(1, buffer_B);

		cl::Event prof_event;

		queue.enqueueNDRangeKernel(kernel_avg, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);

		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size_float, &B[0], NULL, &b_event);

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;

		std::cout << "Kernel execution time [ms (ns)]: " <<
		(float(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()) -
		float(prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())) / float(1000000) << "ms (" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns)" << std::endl;
		std::cout << "Vector A write time [ms (ns)]: " << (float(a_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()) -
		float(a_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())) / float(1000000) << "ms (" << a_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		a_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns)" << std::endl;
		std::cout << "Vector B write time [ms (ns)]: " << (float(b_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()) -
		float(b_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())) / float(1000000) << "ms (" << b_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		b_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << "ns)" << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}