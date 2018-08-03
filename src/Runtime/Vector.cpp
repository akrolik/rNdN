#include "Runtime/Vector.h"

#include "Utils/Logger.h"

namespace Runtime {

Vector *Vector::CreateVector(const HorseIR::PrimitiveType *type, unsigned long size)
{
	switch (type->GetKind())
	{
		case HorseIR::PrimitiveType::Kind::Bool:
			return new TypedVector<uint8_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int8:
			return new TypedVector<int8_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int16:
			return new TypedVector<int16_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int32:
			return new TypedVector<int32_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int64:
			return new TypedVector<int64_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Float32:
			return new TypedVector<float>(type, size);
		case HorseIR::PrimitiveType::Kind::Float64:
			return new TypedVector<double>(type, size);
		default:   
			Utils::Logger::LogError("Unable to create vector of type " + type->ToString());
	}
}

// void Vector::Dump() const
// {
// 	//TODO:
// }

}
