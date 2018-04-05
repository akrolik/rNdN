#include "GPUUtil/CUDAKernelInvocation.h"

#include "GPUUtil/CUDAUtils.h"

CUDAKernelInvocation::CUDAKernelInvocation(CUDAKernel& kernel) : m_kernel(kernel)
{
	m_parameters = ::operator new(m_kernel.GetParametersCount());
}

CUDAKernelInvocation::~CUDAKernelInvocation()
{
	::operator delete(m_parameters);
}

void CUDAKernelInvocation::SetParam(unsigned int index, CUDABuffer &buffer)
{
	((void **)m_parameters)[index] = &buffer.GetGPUBuffer();

	// checkDriverResult(cuParamSetv(m_kernel.GetKernel(), m_paramSize, &buffer.GetGPUBuffer(), buffer.GetSize()));
	// m_paramSize += buffer.GetSize();
}

void CUDAKernelInvocation::Launch()
{
	checkDriverResult(cuLaunchKernel(
				m_kernel.GetKernel(),
				m_blocksX, m_blocksY, m_blocksZ,
				m_shapeX, m_shapeY, m_shapeZ,
				0, 0, (void **)m_parameters, 0
	));

	std::cout << "Kernel '" << m_kernel.GetName() << "' launched" << std::endl;
}
