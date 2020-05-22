#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include <string>
#include <vector>

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Tree/Tree.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

class ListBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::List;

	static ListBuffer *CreateEmpty(const HorseIR::ListType *type, const Analysis::ListShape *shape);

	ListBuffer(DataBuffer *cell) : ListBuffer(std::vector<DataBuffer *>({cell})) {}
	ListBuffer(const std::vector<DataBuffer *>& cells);
	~ListBuffer() override;
	
	ListBuffer *Clone() const override
	{
		std::vector<DataBuffer *> cells;
		for (const auto cell : m_cells)
		{
			cells.push_back(cell->Clone());
		}
		return new ListBuffer(cells);
	}

	// Type/Shape

	const HorseIR::ListType *GetType() const override { return m_type; }
	const Analysis::ListShape *GetShape() const override { return m_shape; }

	// Cells

	const std::vector<DataBuffer *>& GetCells() const { return m_cells; }
	DataBuffer *GetCell(unsigned int index) const { return m_cells.at(index); }
	size_t GetCellCount() const { return m_cells.size(); }

	// Sizing

	void ResizeCells(unsigned int size)
	{
		auto oldDescription = Description();
		auto changed = false;

		for (auto cell : m_cells)
		{
			if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(cell, false))
			{
				changed |= vectorBuffer->Resize(size);
			}
			else
			{
				Utils::Logger::LogError("List buffer resize may only apply vector cells, received " + cell->Description());
			}
		}

		if (changed)
		{
			// Invalidate the GPU content as the cell buffers may have been reallocated

			InvalidateGPU();

			// Propagate shape change

			delete m_shape;

			std::vector<const Analysis::Shape *> cellShapes;
			for (const auto& cell : m_cells)
			{
				cellShapes.push_back(cell->GetShape());
			}
			m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(m_cells.size()), cellShapes);

			if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
			{
				Utils::Logger::LogDebug("Resized list buffer [" + oldDescription + "] to [" + Description() + "]");
			}
		}
	}

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

	CUDA::Buffer *GetGPUSizeBuffer() const override
	{
		ValidateGPU();
		return m_gpuSizeBuffer;
	}

	bool ReallocateGPUBuffer() override
	{
		// Resize all cells in the list individually

		auto oldDescription = Description();
		auto changed = false;

		for (auto cell : m_cells)
		{
			changed |= cell->ReallocateGPUBuffer();
		}

		if (changed)
		{
			// Invalidate the GPU content as the cell buffers may have been reallocated

			InvalidateGPU();

			// Propagate shape change

			delete m_shape;

			std::vector<const Analysis::Shape *> cellShapes;
			for (const auto& cell : m_cells)
			{
				cellShapes.push_back(cell->GetShape());
			}
			m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(m_cells.size()), cellShapes);

			if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
			{
				Utils::Logger::LogDebug("Resized list buffer [" + oldDescription + "] to [" + Description() + "]");
			}
		}
		return changed;
	}

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override
	{
		for (auto i = 0u; i < m_cells.size(); ++i)
		{
			m_cells.at(i)->Clear(mode);
		}
	}

private:
	bool IsAllocatedOnCPU() const override { return true; }
	bool IsAllocatedOnGPU() const override { return (m_gpuBuffer != nullptr); }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override
	{
		auto cellCount = m_cells.size();
		size_t bufferSize = cellCount * sizeof(CUdeviceptr);

		m_gpuBuffer = new CUDA::Buffer(bufferSize);
		m_gpuBuffer->AllocateOnGPU();

		m_gpuSizeBuffer = new CUDA::Buffer(bufferSize);
		m_gpuSizeBuffer->AllocateOnGPU();
	}

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override
	{
		auto cellCount = m_cells.size();
		size_t bufferSize = cellCount * sizeof(CUdeviceptr);

		if (m_gpuDataPointers == nullptr)
		{
			m_gpuDataPointers = new CUdeviceptr[bufferSize];
		}

		if (m_gpuSizePointers == nullptr)
		{
			m_gpuSizePointers = new CUdeviceptr[bufferSize];
		}

		for (auto i = 0u; i < cellCount; ++i)
		{
			auto buffer = m_cells.at(i);
			if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(buffer, false))
			{
				m_gpuDataPointers[i] = vectorBuffer->GetGPUReadBuffer()->GetGPUBuffer();
				m_gpuSizePointers[i] = vectorBuffer->GetGPUSizeBuffer()->GetGPUBuffer();
			}
			else
			{
				Utils::Logger::LogError("GPU list buffers may only have vector cells, received " + buffer->Description());
			}
		}

		// Data

		m_gpuBuffer->SetCPUBuffer(m_gpuDataPointers);
		m_gpuBuffer->TransferToGPU();

		// Size

		m_gpuSizeBuffer->SetCPUBuffer(m_gpuSizePointers);
		m_gpuSizeBuffer->TransferToGPU();
	}

	HorseIR::ListType *m_type = nullptr;
	Analysis::ListShape *m_shape = nullptr;

	std::vector<DataBuffer *> m_cells;

	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuSizeBuffer = nullptr;

	mutable CUdeviceptr *m_gpuDataPointers = nullptr;
	mutable CUdeviceptr *m_gpuSizePointers = nullptr;
};

}
