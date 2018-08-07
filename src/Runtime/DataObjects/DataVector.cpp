#include "Runtime/DataObjects/DataVector.h"

#include "Utils/Logger.h"

namespace Runtime {

DataVector *DataVector::CreateVector(const HorseIR::BasicType *type, unsigned long size)
{
	switch (type->GetKind())
	{
		case HorseIR::BasicType::Kind::Bool:
			return new TypedDataVector<uint8_t>(type, size);
		case HorseIR::BasicType::Kind::Int8:
			return new TypedDataVector<int8_t>(type, size);
		case HorseIR::BasicType::Kind::Int16:
			return new TypedDataVector<int16_t>(type, size);
		case HorseIR::BasicType::Kind::Int32:
			return new TypedDataVector<int32_t>(type, size);
		case HorseIR::BasicType::Kind::Int64:
			return new TypedDataVector<int64_t>(type, size);
		case HorseIR::BasicType::Kind::Float32:
			return new TypedDataVector<float>(type, size);
		case HorseIR::BasicType::Kind::Float64:
			return new TypedDataVector<double>(type, size);
		case HorseIR::BasicType::Kind::Symbol:
			return new TypedDataVector<std::string>(type, size);
		case HorseIR::BasicType::Kind::String:
			return new TypedDataVector<std::string>(type, size);
		default:   
			Utils::Logger::LogError("Unable to create vector of type " + type->ToString());
	}
}

// void Vector::Dump() const
// {
// 	//TODO:
// }

}
