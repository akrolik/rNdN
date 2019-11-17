#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <vector>

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Logger.h"

namespace Runtime {

class ListBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::List;

	static ListBuffer *CreateEmpty(const HorseIR::ListType *type, const Analysis::ListShape *shape);

	ListBuffer(DataBuffer *cell) : ListBuffer(std::vector<DataBuffer *>({cell})) {}
	ListBuffer(const std::vector<DataBuffer *>& cells);
	~ListBuffer() override;
	
	// Type/Shape

	const HorseIR::ListType *GetType() const override { return m_type; }
	const Analysis::ListShape *GetShape() const override { return m_shape; }

	// Cells

	const std::vector<DataBuffer *>& GetCells() { return m_cells; }
	DataBuffer *GetCell(unsigned int index) { return m_cells.at(index); }
	size_t GetCellCount() const { return m_cells.size(); }

	// CPU/GPU management

	CUDA::Buffer *GetGPUWriteBuffer() override
	{
		ValidateGPU();
		for (auto i = 0u; i < m_cells.size(); ++i)
		{
			m_cells.at(i)->GetGPUWriteBuffer();
		}
		return m_gpuBuffer;
	}

	CUDA::Buffer *GetGPUReadBuffer() const override
	{
		ValidateGPU();
		for (auto i = 0u; i < m_cells.size(); ++i)
		{
			m_cells.at(i)->GetGPUReadBuffer();
		}
		return m_gpuBuffer;
	}

	size_t GetGPUBufferSize() const override
	{
		return (sizeof(CUdeviceptr) * m_cells.size());
	}

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	bool IsAllocatedOnCPU() const { return true; }
	bool IsAllocatedOnGPU() const { return (m_gpuBuffer != nullptr); }

	void AllocateCPUBuffer() const {} // Do nothing
	void AllocateGPUBuffer() const
	{
		m_gpuBuffer = new CUDA::Buffer(sizeof(CUdeviceptr) * m_cells.size());
		m_gpuBuffer->AllocateOnGPU();
	}

	void ValidateCPU() const { m_cpuConsistent = true; } // Always consistent
	void ValidateGPU() const
	{
		if (!m_gpuConsistent)
		{
			// Only allocate on GPU once - device pointers may never change

			if (!IsAllocatedOnGPU())
			{
				AllocateGPUBuffer();

				auto cellCount = m_cells.size();
				size_t bufferSize = cellCount * sizeof(CUdeviceptr);

				void *devicePointers = malloc(bufferSize);
				CUdeviceptr *data = reinterpret_cast<CUdeviceptr *>(devicePointers);

				for (auto i = 0u; i < cellCount; ++i)
				{
					auto buffer = m_cells.at(i);
					if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(buffer))
					{
						data[i] = buffer->GetGPUReadBuffer()->GetGPUBuffer();
					}
					else
					{
						Utils::Logger::LogError("GPU list buffers may only have vector cells, received " + buffer->Description());
					}
				}

				m_gpuBuffer->SetCPUBuffer(devicePointers);
				m_gpuBuffer->TransferToGPU();
			}

			m_gpuConsistent = true;
		}
	}
	HorseIR::ListType *m_type = nullptr;
	Analysis::ListShape *m_shape = nullptr;

	std::vector<DataBuffer *> m_cells;
	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
};

}
