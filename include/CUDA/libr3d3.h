#pragma once

#include <memory>

#include "CUDA/Device.h"

#include "PTX/PTX.h"

namespace CUDA {

class libr3d3
{
public:
	static PTX::Program *CreateProgram(const std::unique_ptr<Device>& device);

private:
	template<PTX::Bits B, class T>
	static void CreateFunction_set(PTX::Module *module, const std::string& typeName);
};

}
