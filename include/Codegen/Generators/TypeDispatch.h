#pragma once

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

// @method Dispatch<G, N> (G = Generator, N = NodeType)
//
// @requires
// 	- G is a class containing template method
// 		template<class T>
// 		void Generate(const N*)
//
// Convenience method for converting between HorseIR dynamic types and PTX static types, and
// instantiating the statically typed PTX generator.
//
// This allows the code generator to centralize the type mapping for various constructs
// (assignments, returns, parameters, expressions) in a single place.

namespace Codegen {

template<class G, class N>
static void Dispatch(G &generator, const HorseIR::Type *type, N *node)
{
	switch (type->GetKind())
	{
		case HorseIR::Type::Kind::Primitive:
			Dispatch<G, N>(generator, static_cast<const HorseIR::PrimitiveType *>(type), node);
			break;
		case HorseIR::Type::Kind::List:
			Dispatch<G, N>(generator, static_cast<const HorseIR::ListType *>(type), node);
			break;
		default:
			std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << generator.m_builder->GetContextString("Dispatch") << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

template<class G, class N>
static void Dispatch(G &generator, const HorseIR::PrimitiveType *type, N *node)
{
	switch (type->GetKind())
	{
		case HorseIR::PrimitiveType::Kind::Bool:
			generator.template Generate<PTX::PredicateType>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int8:
			generator.template Generate<PTX::Int8Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int16:
			generator.template Generate<PTX::Int16Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int32:
			generator.template Generate<PTX::Int32Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int64:
			generator.template Generate<PTX::Int64Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Float32:
			generator.template Generate<PTX::Float32Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Float64:
			generator.template Generate<PTX::Float64Type>(node);
			break;
		default:
			std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << generator.m_builder->GetContextString("Dispatch") << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

template<class G, class N>
static void Dispatch(G &generator, const HorseIR::ListType *type, N *node)
{
	Dispatch<G, N>(generator, static_cast<const HorseIR::PrimitiveType *>(type->GetElementType()), node);
}

}
