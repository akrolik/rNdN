#pragma once

#include <memory>

#include "CUDA/Device.h"
#include "CUDA/ExternalModule.h"

namespace CUDA {

class libdevice
{
public:
	static ExternalModule CreateModule(const std::unique_ptr<Device>& device);
};

}
