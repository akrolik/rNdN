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

	const std::vector<DataBuffer *>& GetCells() const { return m_cells; }
	DataBuffer *GetCell(unsigned int index) const { return m_cells.at(index); }
	size_t GetCellCount() const { return m_cells.size(); }

	// CPU/GPU management

	void ValidateCPU(bool recursive = false) const override
	{
		DataBuffer::ValidateCPU(recursive);
		if (recursive)
		{
			for (const auto buffer : m_cells)
			{
				buffer->ValidateCPU(true);
			}
		}
	}

	void ValidateGPU(bool recursive = false) const override
	{
		DataBuffer::ValidateGPU(recursive);
		if (recursive)
		{
			for (const auto buffer : m_cells)
			{
				buffer->ValidateGPU(true);
			}
		}
	}

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

	void SetGPUSizeBuffer(CUDA::Buffer *sizeBuffer) { m_sizeBuffer = sizeBuffer; }
	CUDA::Buffer *GetGPUSizeBuffer() const { return m_sizeBuffer; }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

	// Clear

	void Clear() override
	{
		for (auto i = 0u; i < m_cells.size(); ++i)
		{
			m_cells.at(i)->Clear();
		}
	}

private:
	bool IsAllocatedOnCPU() const override { return true; }
	bool IsAllocatedOnGPU() const override { return (m_gpuBuffer != nullptr); }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override
	{
		m_gpuBuffer = new CUDA::Buffer(sizeof(CUdeviceptr) * m_cells.size());
		m_gpuBuffer->AllocateOnGPU();
	}

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override
	{
		auto cellCount = m_cells.size();
		size_t bufferSize = cellCount * sizeof(CUdeviceptr);

		void *devicePointers = malloc(bufferSize);
		CUdeviceptr *data = reinterpret_cast<CUdeviceptr *>(devicePointers);

		for (auto i = 0u; i < cellCount; ++i)
		{
			auto buffer = m_cells.at(i);
			if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(buffer, false))
			{
				data[i] = vectorBuffer->GetGPUReadBuffer()->GetGPUBuffer();
			}
			else
			{
				Utils::Logger::LogError("GPU list buffers may only have vector cells, received " + buffer->Description());
			}
		}

		m_gpuBuffer->SetCPUBuffer(devicePointers);
		m_gpuBuffer->TransferToGPU();
	}

	HorseIR::ListType *m_type = nullptr;
	Analysis::ListShape *m_shape = nullptr;

	std::vector<DataBuffer *> m_cells;
	mutable CUDA::Buffer *m_gpuBuffer = nullptr;

	CUDA::Buffer *m_sizeBuffer = nullptr;
};

}
