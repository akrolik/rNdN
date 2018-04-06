#include "CUDA/KernelInvocation.h"

#include "CUDA/Utils.h"

namespace CUDA {

KernelInvocation::KernelInvocation(Kernel& kernel) : m_kernel(kernel)
{
	m_parameters = ::operator new(m_kernel.GetParametersCount());
}

KernelInvocation::~KernelInvocation()
{
	::operator delete(m_parameters);
}

void KernelInvocation::SetParam(unsigned int index, Buffer &buffer)
{
	((void **)m_parameters)[index] = &buffer.GetGPUBuffer();

	// checkDriverResult(cuParamSetv(m_kernel.GetKernel(), m_paramSize, &buffer.GetGPUBuffer(), buffer.GetSize()));
	// m_paramSize += buffer.GetSize();
}

void KernelInvocation::Launch()
{
	checkDriverResult(cuLaunchKernel(
				m_kernel.GetKernel(),
				m_blocksX, m_blocksY, m_blocksZ,
				m_shapeX, m_shapeY, m_shapeZ,
				0, 0, (void **)m_parameters, 0
	));

	std::cout << "Kernel '" << m_kernel.GetName() << "' launched" << std::endl;
}

}
