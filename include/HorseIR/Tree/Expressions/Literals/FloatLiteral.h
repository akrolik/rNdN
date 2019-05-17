#pragma once

#include "HorseIR/Analysis/PrettyPrinter.h"
#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/Float32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Float64Literal.h"
#include "HorseIR/Tree/Types/BasicType.h"

#include "Utils/Logger.h"

namespace HorseIR {

template<typename T>
static std::vector<T> ConvertValues(const std::vector<double>& values)
{
	std::vector<T> converted;
	for (double value : values)
	{
		converted.push_back(static_cast<T>(value));
	}
	return converted;
}

template<typename S>
static VectorLiteral *CreateFloatLiteral(const std::vector<S>& values, BasicType *type)
{
	switch (type->GetBasicKind())
	{
		case BasicType::BasicKind::Float32:
			return new Float32Literal(ConvertValues<float>(values));
		case BasicType::BasicKind::Float64:
			return new Float64Literal(ConvertValues<double>(values));
		default:
			Utils::Logger::LogError("Invalid type '" + PrettyPrinter::PrintString(type) + "' for float literal");
	}
}

template<typename S>
static VectorLiteral *CreateFloatLiteral(S value, BasicType *type)
{
	std::vector<S> values = { value };
	return CreateFloatLiteral<S>(values, type);
}

}
