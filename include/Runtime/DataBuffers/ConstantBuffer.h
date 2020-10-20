#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include "CUDA/Constant.h"

#include "HorseIR/Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

class BufferUtils;
class ConstantBuffer : public DataBuffer
{
	friend class BufferUtils;
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Constant;

	~ConstantBuffer() override
	{
		delete m_type;
		delete m_shape;
	}

	virtual ConstantBuffer *Clone() const = 0;

	// Type/Shape

	const HorseIR::BasicType *GetType() const override { return m_type; }
	const HorseIR::Analysis::VectorShape *GetShape() const override { return m_shape; }

	// Clear, nothing to do

	void Clear(ClearMode mode = ClearMode::Zero) override {}

protected:
	ConstantBuffer(const HorseIR::BasicType::BasicKind basicKind) : DataBuffer(DataBuffer::Kind::Constant)
	{
		m_type = new HorseIR::BasicType(basicKind);
		m_shape = new HorseIR::Analysis::VectorShape(new HorseIR::Analysis::Shape::ConstantSize(1));
	}

	const HorseIR::BasicType *m_type = nullptr;
	const HorseIR::Analysis::VectorShape *m_shape = nullptr;
};

template<typename T>
class TypedConstantBuffer : public ConstantBuffer
{
public:
	TypedConstantBuffer(const HorseIR::BasicType::BasicKind basicKind, const T& value) : ConstantBuffer(basicKind), m_value(value) {}
	~TypedConstantBuffer() override
	{
		delete m_gpuBuffer;
	}

	TypedConstantBuffer<T> *Clone() const override
	{
		return new TypedConstantBuffer<T>(m_type->GetBasicKind(), m_value);
	}

	// GPU buffers

	CUDA::Constant *GetGPUReadBuffer() const override
	{
		ValidateGPU();
		return m_gpuBuffer;
	}

	CUDA::Constant *GetGPUWriteBuffer() override
	{
		Utils::Logger::LogError("Constant buffers are read-only");
	}

private:
	bool IsAllocatedOnCPU() const override { return true; }
	bool IsAllocatedOnGPU() const override { return (m_gpuBuffer != nullptr); }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override
	{
		m_gpuBuffer = new CUDA::TypedConstant(m_value);
	}

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override {} // Not needed

	std::string Description() const override
	{
		return std::to_string(m_value) + ":" + HorseIR::PrettyPrinter::PrettyString(m_type);
	}

	std::string DebugDump() const override
	{
		return std::to_string(m_value) + ":" + HorseIR::PrettyPrinter::PrettyString(m_type);
	}

protected:
	T m_value;
	mutable CUDA::TypedConstant<T> *m_gpuBuffer = nullptr;
};

}
