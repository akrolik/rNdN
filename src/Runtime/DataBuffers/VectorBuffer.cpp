#include "Runtime/DataBuffers/VectorBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

VectorBuffer *VectorBuffer::Create(const HorseIR::BasicType *type, unsigned long size)
{
	switch (type->GetBasicKind())
	{
		case HorseIR::BasicType::BasicKind::Boolean:
			return new TypedVectorBuffer<uint8_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int8:
			return new TypedVectorBuffer<int8_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int16:
			return new TypedVectorBuffer<int16_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int32:
			return new TypedVectorBuffer<int32_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int64:
			return new TypedVectorBuffer<int64_t>(type, size);
		case HorseIR::BasicType::BasicKind::Float32:
			return new TypedVectorBuffer<float>(type, size);
		case HorseIR::BasicType::BasicKind::Float64:
			return new TypedVectorBuffer<double>(type, size);
		case HorseIR::BasicType::BasicKind::Symbol:
			return new TypedVectorBuffer<std::string>(type, size);
		case HorseIR::BasicType::BasicKind::String:
			return new TypedVectorBuffer<std::string>(type, size);
		default:   
			Utils::Logger::LogError("Unable to create vector of type " + HorseIR::PrettyPrinter::PrettyString(type));
	}
}

}
