#pragma once

#include "HorseIR/Tree/Types/Type.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/BasicType.h"

#include "Utils/Logger.h"

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

template<class G, typename... N>
static void DispatchBasic(G &generator, const HorseIR::BasicType *type, N ...nodes);

template<class G, typename... N>
static void DispatchList(G &generator, const HorseIR::ListType *type, N ...nodes);

template<class G, typename... N>
static void DispatchType(G &generator, const HorseIR::Type *type, N ...nodes)
{
	switch (type->GetKind())
	{
		case HorseIR::Type::Kind::Basic:
			DispatchBasic<G, N...>(generator, static_cast<const HorseIR::BasicType *>(type), nodes...);
			break;
		case HorseIR::Type::Kind::List:
			DispatchList<G, N...>(generator, static_cast<const HorseIR::ListType *>(type), nodes...);
			break;
		default:
			Utils::Logger::LogError("Unsupported type " + type->ToString() + " in function " + generator.m_builder.GetContextString("Dispatch"));
	}
}

template<class G, typename... N>
static void DispatchBasic(G &generator, const HorseIR::BasicType *type, N ...nodes)
{
	switch (type->GetKind())
	{
		case HorseIR::BasicType::Kind::Bool:
			generator.template Generate<PTX::PredicateType>(nodes...);
			break;
		case HorseIR::BasicType::Kind::Int8:
			generator.template Generate<PTX::Int8Type>(nodes...);
			break;
		case HorseIR::BasicType::Kind::Int16:
			generator.template Generate<PTX::Int16Type>(nodes...);
			break;
		case HorseIR::BasicType::Kind::Int32:
			generator.template Generate<PTX::Int32Type>(nodes...);
			break;
		case HorseIR::BasicType::Kind::Int64:
			generator.template Generate<PTX::Int64Type>(nodes...);
			break;
		case HorseIR::BasicType::Kind::Float32:
			generator.template Generate<PTX::Float32Type>(nodes...);
			break;
		case HorseIR::BasicType::Kind::Float64:
			generator.template Generate<PTX::Float64Type>(nodes...);
			break;
		default:
			Utils::Logger::LogError("Unsupported type " + type->ToString() + " in function " + generator.m_builder.GetContextString("Dispatch"));
	}
}

template<class G, typename... N>
static void DispatchList(G &generator, const HorseIR::ListType *type, N ...nodes)
{
	DispatchBasic<G, N...>(generator, static_cast<const HorseIR::BasicType *>(type->GetElementType()), nodes...);
}

}
