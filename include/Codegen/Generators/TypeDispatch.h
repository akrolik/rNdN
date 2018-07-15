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

template<class G, class... N>
static void DispatchPrimitive(G &generator, const HorseIR::PrimitiveType *type, N* ...nodes);

template<class G, class... N>
static void DispatchList(G &generator, const HorseIR::ListType *type, N* ...nodes);

template<class G, class... N>
static void DispatchType(G &generator, const HorseIR::Type *type, N* ...nodes)
{
	switch (type->GetKind())
	{
		case HorseIR::Type::Kind::Primitive:
			DispatchPrimitive<G, N...>(generator, static_cast<const HorseIR::PrimitiveType *>(type), nodes...);
			break;
		case HorseIR::Type::Kind::List:
			DispatchList<G, N...>(generator, static_cast<const HorseIR::ListType *>(type), nodes...);
			break;
		default:
			std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << generator.m_builder->GetContextString("Dispatch") << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

template<class G, class... N>
static void DispatchPrimitive(G &generator, const HorseIR::PrimitiveType *type, N* ...nodes)
{
	switch (type->GetKind())
	{
		case HorseIR::PrimitiveType::Kind::Bool:
			generator.template Generate<PTX::PredicateType>(nodes...);
			break;
		case HorseIR::PrimitiveType::Kind::Int8:
			generator.template Generate<PTX::Int8Type>(nodes...);
			break;
		case HorseIR::PrimitiveType::Kind::Int16:
			generator.template Generate<PTX::Int16Type>(nodes...);
			break;
		case HorseIR::PrimitiveType::Kind::Int32:
			generator.template Generate<PTX::Int32Type>(nodes...);
			break;
		case HorseIR::PrimitiveType::Kind::Int64:
			generator.template Generate<PTX::Int64Type>(nodes...);
			break;
		case HorseIR::PrimitiveType::Kind::Float32:
			generator.template Generate<PTX::Float32Type>(nodes...);
			break;
		case HorseIR::PrimitiveType::Kind::Float64:
			generator.template Generate<PTX::Float64Type>(nodes...);
			break;
		default:
			std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << generator.m_builder->GetContextString("Dispatch") << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

template<class G, class... N>
static void DispatchList(G &generator, const HorseIR::ListType *type, N* ...nodes)
{
	DispatchPrimitive<G, N...>(generator, static_cast<const HorseIR::PrimitiveType *>(type->GetElementType()), nodes...);
}

}
