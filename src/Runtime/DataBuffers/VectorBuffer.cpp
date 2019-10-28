#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

VectorBuffer *VectorBuffer::Create(const HorseIR::BasicType *type, const Analysis::VectorShape *shape)
{
	const auto size = shape->GetSize();
	if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(size))
	{
		auto value = constantSize->GetValue();
		switch (type->GetBasicKind())
		{
			case HorseIR::BasicType::BasicKind::Boolean:
				return new TypedVectorBuffer<uint8_t>(type, value);
			case HorseIR::BasicType::BasicKind::Int8:
				return new TypedVectorBuffer<int8_t>(type, value);
			case HorseIR::BasicType::BasicKind::Int16:
				return new TypedVectorBuffer<int16_t>(type, value);
			case HorseIR::BasicType::BasicKind::Int32:
				return new TypedVectorBuffer<int32_t>(type, value);
			case HorseIR::BasicType::BasicKind::Int64:
				return new TypedVectorBuffer<int64_t>(type, value);
			case HorseIR::BasicType::BasicKind::Float32:
				return new TypedVectorBuffer<float>(type, value);
			case HorseIR::BasicType::BasicKind::Float64:
				return new TypedVectorBuffer<double>(type, value);
			case HorseIR::BasicType::BasicKind::Symbol:
				return new TypedVectorBuffer<std::string>(type, value);
			case HorseIR::BasicType::BasicKind::String:
				return new TypedVectorBuffer<std::string>(type, value);
			default:   
				Utils::Logger::LogError("Unable to create vector of type " + HorseIR::PrettyPrinter::PrettyString(type));
		}
	}
	Utils::Logger::LogError("Vector buffer expects constant size");
}

VectorBuffer::~VectorBuffer()
{
	delete m_type;
	delete m_shape;
}

}
