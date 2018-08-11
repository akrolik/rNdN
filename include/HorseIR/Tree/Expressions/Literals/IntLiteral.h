#pragma once

#include <cstdint>

#include "HorseIR/Tree/Expressions/Expression.h"
#include "HorseIR/Tree/Expressions/Literals/Int8Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int16Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int32Literal.h"
#include "HorseIR/Tree/Expressions/Literals/Int64Literal.h"
#include "HorseIR/Tree/Types/BasicType.h"

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

static Expression *CreateIntLiteral(const std::vector<std::int64_t>& values, BasicType *type)
{
	switch (type->GetKind())
	{
		case BasicType::Kind::Int8:
			return new Int8Literal(ConvertValues<std::int8_t>(values));
		case BasicType::Kind::Int16:
			return new Int16Literal(ConvertValues<std::int16_t>(values));
		case BasicType::Kind::Int32:
			return new Int32Literal(ConvertValues<std::int32_t>(values));
		case BasicType::Kind::Int64:
			return new Int64Literal(ConvertValues<std::int64_t>(values));
		default:
			Utils::Logger::LogError("Invalid type '" + type->ToString() + "' for integer literal");
	}
}

static Expression *CreateIntLiteral(std::int64_t value, BasicType *type)
{
	std::vector<std::int64_t> values = { value };
	return CreateIntLiteral(values, type);
}

}
