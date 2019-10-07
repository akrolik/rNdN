#include "CUDA/Buffer.h"

#include "CUDA/Chrono.h"
#include "CUDA/Utils.h"

#include "Utils/Logger.h"

namespace CUDA {

void Buffer::AllocateOnGPU()
{
	auto start = Chrono::Start();

	auto paddedSize = GetPaddedSize();
	checkDriverResult(cuMemAlloc(&m_GPUBuffer, paddedSize));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA allocation (" + std::to_string(paddedSize) + " bytes)", time);
}

void Buffer::TransferToGPU()
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemcpyHtoD(m_GPUBuffer, m_CPUBuffer, m_size));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA transfer (" + std::to_string(m_size) + " bytes) ->", time);
}

void Buffer::TransferToCPU()
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemcpyDtoH(m_CPUBuffer, m_GPUBuffer, m_size));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA transfer (" + std::to_string(m_size) + " bytes) <-", time);
}

}
