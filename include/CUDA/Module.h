#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDA {

class Module
{
public:
	Module(void *binary, std::size_t binarySize);

	void *GetBinary() { return m_binary; }
	std::size_t GetBinarySize() const { return m_binarySize; }

	const CUmodule& GetModule() const { return m_module; }

private:
	void *m_binary = nullptr;
	std::size_t m_binarySize = 0;

	CUmodule m_module;
};

}
