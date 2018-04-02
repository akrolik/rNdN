#include <iostream>
#include "GPUUtil/CUDA.h"

int main(int argc, char *argv[])
{
	CUDA c;
	c.Initialize();

	if (c.GetDeviceCount() == 0)
	{
		std::cerr << "[Error] No connected devices detected" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	CUDADevice device = c.GetDevice(0);
	device.SetActive();
}
