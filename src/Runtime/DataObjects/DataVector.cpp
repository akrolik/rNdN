#include "Runtime/DataObjects/DataVector.h"

#include "Utils/Logger.h"

namespace Runtime {

DataVector *DataVector::CreateVector(const HorseIR::PrimitiveType *type, unsigned long size)
{
	switch (type->GetKind())
	{
		case HorseIR::PrimitiveType::Kind::Bool:
			return new TypedDataVector<uint8_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int8:
			return new TypedDataVector<int8_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int16:
			return new TypedDataVector<int16_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int32:
			return new TypedDataVector<int32_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Int64:
			return new TypedDataVector<int64_t>(type, size);
		case HorseIR::PrimitiveType::Kind::Float32:
			return new TypedDataVector<float>(type, size);
		case HorseIR::PrimitiveType::Kind::Float64:
			return new TypedDataVector<double>(type, size);
		default:   
			Utils::Logger::LogError("Unable to create vector of type " + type->ToString());
	}
}

// void Vector::Dump() const
// {
// 	//TODO:
// }

}
