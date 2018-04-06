#include "CUDA/Buffer.h"

#include "CUDA/Utils.h"

namespace CUDA {

void Buffer::AllocateOnGPU()
{
	checkDriverResult(cuMemAlloc(&m_GPUBuffer, m_size));
}

void Buffer::TransferToGPU()
{
	checkDriverResult(cuMemcpyHtoD(m_GPUBuffer, m_CPUBuffer, m_size));
}

void Buffer::TransferToCPU()
{
	checkDriverResult(cuMemcpyDtoH(m_CPUBuffer, m_GPUBuffer, m_size));
}

}
