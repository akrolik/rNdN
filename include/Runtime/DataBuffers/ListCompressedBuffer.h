#pragma once

#include "Runtime/DataBuffers/ListBuffer.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class ListCompressedBuffer : public ListBuffer
{
public:
	ListCompressedBuffer(const TypedVectorBuffer<CUdeviceptr> *dataAddresses, const TypedVectorBuffer<CUdeviceptr> *sizeAddresses, const TypedVectorBuffer<std::int32_t> *sizes, VectorBuffer *values);
	~ListCompressedBuffer() override;
	
	ListCompressedBuffer *Clone() const override;

	// Cells

	const std::vector<DataBuffer *>& GetCells() const override { return m_cells; }
	DataBuffer *GetCell(unsigned int index) const override { return m_cells.at(index); }
	size_t GetCellCount() const override { return m_dataAddresses->GetElementCount(); }

	// Sizing

	void ResizeCells(unsigned int size) override {} // Do nothing

	// CPU/GPU management

	void ValidateCPU(bool recursive = false) const override;
	void ValidateGPU(bool recursive = false) const override;

	CUDA::Buffer *GetGPUWriteBuffer() override;
	CUDA::Buffer *GetGPUReadBuffer() const override;

	size_t GetGPUBufferSize() const override { return m_dataAddresses->GetGPUBufferSize(); }
	CUDA::Buffer *GetGPUSizeBuffer() const override
	{
		//TODO: Move this
		m_sizes->ValidateGPU();
		return m_sizeAddresses->GetGPUReadBuffer();
	}

	bool ReallocateGPUBuffer() override { return false; } // Do nothing

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	//TODO: Make these functions below correct
	bool IsAllocatedOnCPU() const override { return true; } // Always allocated on CPU
	bool IsAllocatedOnGPU() const override { return true; }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override {} // Do nothing

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override {} // Always consistent

	const TypedVectorBuffer<CUdeviceptr> *m_dataAddresses = nullptr;
	const TypedVectorBuffer<CUdeviceptr> *m_sizeAddresses = nullptr;
	const TypedVectorBuffer<std::int32_t> *m_sizes = nullptr;
	VectorBuffer *m_values = nullptr;

	std::vector<DataBuffer *> m_cells;
};

}
