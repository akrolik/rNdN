#pragma once

#include "CUDA/Device.h"
#include "CUDA/ExternalModule.h"

namespace CUDA {

class libdevice
{
public:
	static ExternalModule CreateModule(const Device& device);
};

}
