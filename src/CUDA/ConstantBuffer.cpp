#include "CUDA/ConstantBuffer.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"
#include "Utils/Math.h"

namespace CUDA {

ConstantBuffer::ConstantBuffer(size_t size) : m_size(size)
{
	m_allocatedSize = Utils::Math::RoundUp(size, 1024);
}

ConstantBuffer::ConstantBuffer(CUdeviceptr buffer, size_t size) : m_GPUBuffer(buffer), m_size(size), m_allocatedSize(size)
{
	m_alias = true;
}

std::string ConstantBuffer::ChronoDescription(const std::string& operation, size_t size)
{
	if (m_tag == "")
	{
		return "CUDA " + operation + " (" + std::to_string(size) + " bytes)";
	}
	return "CUDA " + operation + " '" + m_tag + "' (" + std::to_string(size) + " bytes)";
}

ConstantBuffer::~ConstantBuffer()
{
	if (!m_alias)
	{
		auto start = Utils::Chrono::StartCUDA(ChronoDescription("free", m_allocatedSize));

		checkDriverResult(cuMemFree(m_GPUBuffer));

		Utils::Chrono::End(start);
	}
}

void ConstantBuffer::AllocateOnGPU()
{
	if (!m_alias)
	{
		auto start = Utils::Chrono::StartCUDA(ChronoDescription("allocation", m_allocatedSize));

		checkDriverResult(cuMemAlloc(&m_GPUBuffer, m_allocatedSize));

		Utils::Chrono::End(start);
	}
}

void ConstantBuffer::Clear()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("clear", m_allocatedSize));

	checkDriverResult(cuMemsetD8(m_GPUBuffer, 0, m_allocatedSize));

	Utils::Chrono::End(start);
}

void ConstantBuffer::TransferToGPU()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("transfer", m_size) + " ->");

	checkDriverResult(cuMemcpyHtoD(m_GPUBuffer, m_CPUBuffer, m_size));

	Utils::Chrono::End(start);
}

}
