#include "Runtime/DataObjects/DataVector.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace Runtime {

DataVector *DataVector::CreateVector(const HorseIR::BasicType *type, unsigned long size)
{
	switch (type->GetBasicKind())
	{
		case HorseIR::BasicType::BasicKind::Boolean:
			return new TypedDataVector<uint8_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int8:
			return new TypedDataVector<int8_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int16:
			return new TypedDataVector<int16_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int32:
			return new TypedDataVector<int32_t>(type, size);
		case HorseIR::BasicType::BasicKind::Int64:
			return new TypedDataVector<int64_t>(type, size);
		case HorseIR::BasicType::BasicKind::Float32:
			return new TypedDataVector<float>(type, size);
		case HorseIR::BasicType::BasicKind::Float64:
			return new TypedDataVector<double>(type, size);
		case HorseIR::BasicType::BasicKind::Symbol:
			return new TypedDataVector<std::string>(type, size);
		case HorseIR::BasicType::BasicKind::String:
			return new TypedDataVector<std::string>(type, size);
		default:   
			Utils::Logger::LogError("Unable to create vector of type " + HorseIR::PrettyPrinter::PrettyString(type));
	}
}

}
