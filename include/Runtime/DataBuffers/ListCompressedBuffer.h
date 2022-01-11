#pragma once

#include "Runtime/DataBuffers/ListBuffer.h"

#include "Runtime/DataBuffers/VectorBuffer.h"

namespace Runtime {

class ListCompressedBuffer : public ListBuffer
{
public:
	static ListCompressedBuffer *CreateEmpty(const HorseIR::BasicType *type, const HorseIR::Analysis::Shape::RangedSize *size);

	ListCompressedBuffer(const TypedVectorBuffer<std::int64_t> *offsets, VectorBuffer *values);
	ListCompressedBuffer(TypedVectorBuffer<std::int32_t> *sizes, VectorBuffer *values);
	~ListCompressedBuffer() override;
	
	ListCompressedBuffer *Clone() const override;

	// Tag

	void SetTag(const std::string& tag) override;

	// Cells

	std::vector<const DataBuffer *> GetCells() const override;
	std::vector<DataBuffer *>& GetCells() override;

	const DataBuffer *GetCell(unsigned int index) const override;
	DataBuffer *GetCell(unsigned int index) override;

	size_t GetCellCount() const override;

	// Sizing

	void ResizeCells(unsigned int size) override {} // Do nothing

	// CPU/GPU management

	void RequireCPUConsistent(bool exclusive) const override;
	void RequireGPUConsistent(bool exclusive) const override;

	CUDA::Buffer *GetGPUWriteBuffer() override;
	const CUDA::Buffer *GetGPUReadBuffer() const override;

	const CUDA::Buffer *GetGPUSizeBuffer() const override;
	CUDA::Buffer *GetGPUSizeBuffer() override;

	size_t GetGPUBufferSize() const override { return m_dataAddresses->GetGPUBufferSize(); }

	bool ReallocateGPUBuffer() override { return false; } // Do nothing

	// Printers

	std::string Description() const override;
	std::string DebugDump(unsigned int indent = 0, bool preindent = false) const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	// CPU/GPU management

	bool IsAllocatedOnCPU() const override { return true; } // Always allocated on CPU
	bool IsAllocatedOnGPU() const override { return true; }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override {} // Do nothing

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override {} // Always consistent

	// Data

	void AllocateCells() const;

	TypedVectorBuffer<CUdeviceptr> *m_dataAddresses = nullptr;
	TypedVectorBuffer<CUdeviceptr> *m_sizeAddresses = nullptr;
	TypedVectorBuffer<std::int32_t> *m_sizes = nullptr;
	VectorBuffer *m_values = nullptr;

	mutable std::vector<DataBuffer *> m_cells;
};

}
