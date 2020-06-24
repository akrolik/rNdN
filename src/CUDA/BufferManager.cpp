#include "CUDA/BufferManager.h"

#include "CUDA/Utils.h"

#include "Utils/Options.h"
#include "Utils/Logger.h"
#include "Utils/Math.h"

namespace CUDA {

void BufferManager::Initialize()
{
	if (Utils::Options::GetAllocatorKind() == Utils::Options::AllocatorKind::Linear)
	{
		auto timeData_start = Utils::Chrono::Start("Create data pages");

		auto& instance = GetInstance();
		auto& buffers = instance.m_gpuBuffers;

		auto pageSize = Utils::Options::Get<unsigned long long>(Utils::Options::Opt_Data_page_size);
		auto pageCount = Utils::Options::Get<unsigned int>(Utils::Options::Opt_Data_page_count);

		for (auto i = 0u; i < pageCount; ++i)
		{
			auto buffer = new Buffer(pageSize);
			buffer->SetTag("page_" + std::to_string(i));
			buffer->AllocateOnGPU();
			buffer->Clear();
			buffers.push_back(buffer);
		}
		Synchronize();

		Utils::Chrono::End(timeData_start);
	}
}

void BufferManager::Clear()
{
	auto& instance = GetInstance();
	instance.m_page = 0;
	instance.m_sbrk = 0;
}

void BufferManager::Destroy()
{
	auto& instance = GetInstance();
	auto& buffers = instance.m_gpuBuffers;

	for (auto buffer : buffers)
	{
		delete buffer;
	}
	buffers.clear();
}

Buffer *BufferManager::CreateBuffer(size_t size)
{
	switch (Utils::Options::GetAllocatorKind())
	{
		case Utils::Options::AllocatorKind::CUDA:
		{
			return new Buffer(size);
		}
		case Utils::Options::AllocatorKind::Linear:
		{
			auto& instance = GetInstance();

			auto pageBuffer = instance.GetPageBuffer(size);
			auto buffer = new Buffer(pageBuffer->GetGPUBuffer() + instance.m_sbrk, size);

			instance.m_sbrk += Utils::Math::RoundUp(size, 1024);

			return buffer;
		}
	}
	Utils::Logger::LogError("Unknown allocator");
}

ConstantBuffer *BufferManager::CreateConstantBuffer(size_t size)
{
	switch (Utils::Options::GetAllocatorKind())
	{
		case Utils::Options::AllocatorKind::CUDA:
		{
			return new ConstantBuffer(size);
		}
		case Utils::Options::AllocatorKind::Linear:
		{
			auto& instance = GetInstance();

			auto pageBuffer = instance.GetPageBuffer(size);
			auto buffer = new ConstantBuffer(pageBuffer->GetGPUBuffer() + instance.m_sbrk, size);

			instance.m_sbrk += Utils::Math::RoundUp(size, 1024);

			return buffer;
		}
	}
	Utils::Logger::LogError("Unknown allocator");
}

Buffer *BufferManager::GetPageBuffer(size_t size)
{
	// Check that the data fits within a single page

	auto pageSize = Utils::Options::Get<unsigned long long>(Utils::Options::Opt_Data_page_size);
	if (size > pageSize)
	{
		Utils::Logger::LogError("CUDA allocation exceeds page size [" + std::to_string(size) + " > " + std::to_string(pageSize) + "]");
	}

	// Check if the size fits within the current page

	if (size > pageSize - m_sbrk)
	{
		m_page++;
	}

	return m_gpuBuffers.at(m_page);
}

}
