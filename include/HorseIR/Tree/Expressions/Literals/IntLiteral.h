#pragma once

#include <cstdint>

#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Literals/VectorLiteral.h"
#include "HorseIR/Tree/Expressions/Literals/Int8Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int16Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int64Literal.h"
#include "HorseIR/Tree/Types/BasicType.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace HorseIR {

template<typename T>
static std::vector<T> ConvertValues(const std::vector<std::int64_t>& values)
{
	std::vector<T> converted;
	for (std::int64_t value : values)
	{
		converted.push_back(static_cast<T>(value));
	}
	return converted;
}

static VectorLiteral *CreateIntLiteral(const std::vector<std::int64_t>& values, BasicType *type)
{
	switch (type->GetBasicKind())
	{
		case BasicType::BasicKind::Int8:
			return new Int8Literal(ConvertValues<std::int8_t>(values));
		case BasicType::BasicKind::Int16:
			return new Int16Literal(ConvertValues<std::int16_t>(values));
		case BasicType::BasicKind::Int32:
			return new Int32Literal(ConvertValues<std::int32_t>(values));
		case BasicType::BasicKind::Int64:
			return new Int64Literal(ConvertValues<std::int64_t>(values));
		default:
			Utils::Logger::LogError("Invalid type '" + PrettyPrinter::PrettyString(type) + "' for integer literal");
	}
}

static VectorLiteral *CreateIntLiteral(std::int64_t value, BasicType *type)
{
	std::vector<std::int64_t> values = { value };
	return CreateIntLiteral(values, type);
}

}
