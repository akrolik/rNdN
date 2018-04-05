#include "GPUUtil/CUDABuffer.h"

#include "GPUUtil/CUDAUtils.h"

void CUDABuffer::AllocateOnGPU()
{
	checkDriverResult(cuMemAlloc(&m_GPUBuffer, m_size));
}

void CUDABuffer::TransferToGPU()
{
	checkDriverResult(cuMemcpyHtoD(m_GPUBuffer, m_CPUBuffer, m_size));
}

void CUDABuffer::TransferToCPU()
{
	checkDriverResult(cuMemcpyDtoH(m_CPUBuffer, m_GPUBuffer, m_size));
}
