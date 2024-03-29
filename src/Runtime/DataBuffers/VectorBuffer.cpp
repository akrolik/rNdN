#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Analysis/Shape/ShapeUtils.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

VectorBuffer *VectorBuffer::CreateEmpty(const HorseIR::BasicType *type, unsigned int size)
{
	switch (type->GetBasicKind())
	{
		case HorseIR::BasicType::BasicKind::Boolean:
		case HorseIR::BasicType::BasicKind::Char:
		case HorseIR::BasicType::BasicKind::Int8:
			return new TypedVectorBuffer<std::int8_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int16:
			return new TypedVectorBuffer<std::int16_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int32:
			return new TypedVectorBuffer<std::int32_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int64:
			return new TypedVectorBuffer<std::int64_t>(type, size);
		case HorseIR::BasicType::BasicKind::Float32:
			return new TypedVectorBuffer<float>(type, size);
		case HorseIR::BasicType::BasicKind::Float64:
			return new TypedVectorBuffer<double>(type, size);
		case HorseIR::BasicType::BasicKind::Symbol:
		case HorseIR::BasicType::BasicKind::String:
			return new TypedVectorBuffer<std::uint64_t>(type, size);
		case HorseIR::BasicType::BasicKind::Date:
		case HorseIR::BasicType::BasicKind::Month:
		case HorseIR::BasicType::BasicKind::Minute:
		case HorseIR::BasicType::BasicKind::Second:
			return new TypedVectorBuffer<std::int32_t>(type, size);
		case HorseIR::BasicType::BasicKind::Time:
		case HorseIR::BasicType::BasicKind::Datetime:
			return new TypedVectorBuffer<std::int64_t>(type, size);
		default:   
			Utils::Logger::LogError("Unable to create vector of type " + HorseIR::PrettyPrinter::PrettyString(type));
	}
}

VectorBuffer *VectorBuffer::CreateEmpty(const HorseIR::BasicType *type, const HorseIR::Analysis::Shape::Size *size)
{
	if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(size))
	{
		return VectorBuffer::CreateEmpty(type, constantSize->GetValue());
	}
	else if (const auto compressedSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::CompressedSize>(size))
	{
		return VectorBuffer::CreateEmpty(type, compressedSize->GetSize());
	}
	else if (const auto rangedSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::RangedSize>(size))
	{
		auto totalSize = 0u;
		for (auto size : rangedSize->GetValues())
		{
			totalSize += size;
		}
		return VectorBuffer::CreateEmpty(type, totalSize);
	}
	Utils::Logger::LogError("Umable to create vector of size " + HorseIR::Analysis::ShapeUtils::SizeString(size));
}

VectorBuffer::VectorBuffer(const std::type_index &tid, const HorseIR::BasicType *type, unsigned long elementCount) :
	ColumnBuffer(DataBuffer::Kind::Vector), m_typeid(tid), m_elementCount(elementCount)
{
	m_type = type->Clone();
	m_shape = new HorseIR::Analysis::VectorShape(new HorseIR::Analysis::Shape::ConstantSize(m_elementCount));
}

VectorBuffer::~VectorBuffer()
{
	delete m_type;
	delete m_shape;
}

}
