#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <vector>

#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class ListBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::List;

	static ListBuffer *Create(const HorseIR::ListType *type, const Analysis::ListShape *shape);

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

	//TODO: Harmonize with the vector buffer
	CUDA::Buffer *GetGPUWriteBuffer() override
	{
		return GetGPUBuffer(true);
	}
	CUDA::Buffer *GetGPUReadBuffer() const override
	{
		return GetGPUBuffer(false);
	}

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	CUDA::Buffer *GetGPUBuffer(bool write) const
	{
		auto cellCount = m_cells.size();
		size_t bufferSize = cellCount * sizeof(CUdeviceptr);

		void *cpuBuffer = malloc(bufferSize);
		CUdeviceptr *data = reinterpret_cast<CUdeviceptr *>(cpuBuffer);

		for (auto i = 0u; i < cellCount; ++i)
		{
			auto buffer = static_cast<VectorBuffer *>(m_cells.at(i));
			if (write)
			{
				data[i] = buffer->GetGPUWriteBuffer()->GetGPUBuffer();
			}
			else
			{
				data[i] = buffer->GetGPUReadBuffer()->GetGPUBuffer();
			}
		}

		auto gpuBuffer = new CUDA::Buffer(cpuBuffer, bufferSize);
		gpuBuffer->AllocateOnGPU();
		gpuBuffer->TransferToGPU();
		return gpuBuffer;
	}

	HorseIR::ListType *m_type = nullptr;
	Analysis::ListShape *m_shape = nullptr;

	std::vector<DataBuffer *> m_cells;
};

}
