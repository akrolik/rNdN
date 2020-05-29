#pragma once

#include "Runtime/DataBuffers/ListBuffer.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class ListCompressedBuffer : public ListBuffer
{
public:
	ListCompressedBuffer(const TypedVectorBuffer<std::int32_t> *sizes, VectorBuffer *values);
	~ListCompressedBuffer() override;
	
	ListCompressedBuffer *Clone() const override;

	// Cells

	const std::vector<DataBuffer *>& GetCells() const override;
	DataBuffer *GetCell(unsigned int index) const override;
	size_t GetCellCount() const override;

	// Sizing

	void ResizeCells(unsigned int size) override {} // Do nothing

	// CPU/GPU management

	void ValidateCPU(bool recursive = false) const override;
	void ValidateGPU(bool recursive = false) const override;

	CUDA::Buffer *GetGPUWriteBuffer() override;
	CUDA::Buffer *GetGPUReadBuffer() const override;
	CUDA::Buffer *GetGPUSizeBuffer() const override;

	size_t GetGPUBufferSize() const override { return m_dataAddresses->GetGPUBufferSize(); }

	bool ReallocateGPUBuffer() override { return false; } // Do nothing

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	bool IsAllocatedOnCPU() const override { return true; } // Always allocated on CPU
	bool IsAllocatedOnGPU() const override { return true; }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override {} // Do nothing

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override {} // Always consistent

	void AllocateCells() const;

	const TypedVectorBuffer<CUdeviceptr> *m_dataAddresses = nullptr;
	const TypedVectorBuffer<CUdeviceptr> *m_sizeAddresses = nullptr;
	const TypedVectorBuffer<std::int32_t> *m_sizes = nullptr;
	VectorBuffer *m_values = nullptr;

	mutable std::vector<DataBuffer *> m_cells;
};

}
