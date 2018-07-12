#pragma once

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

namespace Codegen {

static HorseIR::PrimitiveType *WidestType(const HorseIR::PrimitiveType *type1, const HorseIR::PrimitiveType *type2)
{
	if (type1->GetKind() == HorseIR::PrimitiveType::Kind::Float64 ||
		type2->GetKind() == HorseIR::PrimitiveType::Kind::Float64)
	{
		return new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Float64);
	}
	else if (type1->GetKind() == HorseIR::PrimitiveType::Kind::Float32 ||
		type2->GetKind() == HorseIR::PrimitiveType::Kind::Float32)
	{
		return new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Float32);
	}
	else if (type1->GetKind() == HorseIR::PrimitiveType::Kind::Int64 ||
		type2->GetKind() == HorseIR::PrimitiveType::Kind::Int64)
	{
		return new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int64);
	}
	else if (type1->GetKind() == HorseIR::PrimitiveType::Kind::Int32 ||
		type2->GetKind() == HorseIR::PrimitiveType::Kind::Int32)
	{
		return new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int32);
	}
	else if (type1->GetKind() == HorseIR::PrimitiveType::Kind::Int16 ||
		type2->GetKind() == HorseIR::PrimitiveType::Kind::Int16)
	{
		return new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int16);
	}
	else if (type1->GetKind() == HorseIR::PrimitiveType::Kind::Int8 ||
		type2->GetKind() == HorseIR::PrimitiveType::Kind::Int8)
	{
		return new HorseIR::PrimitiveType(HorseIR::PrimitiveType::Kind::Int8);
	}
	std::cerr << "[ERROR] Unknown widest type for " << type1->ToString() << " and " << type2->ToString() << std::endl;
	std::exit(EXIT_FAILURE);
}

static HorseIR::PrimitiveType *WidestType(const HorseIR::Type *type1, const HorseIR::Type *type2)
{
	if (type1->GetKind() == HorseIR::Type::Kind::Primitive &&
		type2->GetKind() == HorseIR::Type::Kind::Primitive)
	{
		return WidestType(static_cast<const HorseIR::PrimitiveType *>(type1), static_cast<const HorseIR::PrimitiveType *>(type2));
	}
	std::cerr << "[ERROR] Unknown widest type for " << type1->ToString() << " and " << type2->ToString() << std::endl;
	std::exit(EXIT_FAILURE);
}

}
