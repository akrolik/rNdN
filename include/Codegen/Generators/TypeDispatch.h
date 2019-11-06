#pragma once

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"
#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"

// @function Dispatch<G, N> (G = Generator, N = NodeType)
//
// @requires
// 	- G is a class containing template function
// 		template<class T>
// 		void Generate(const N*)
//
// Convenience function for converting between HorseIR dynamic types and PTX static types, and
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
	switch (type->m_kind)
	{
		case HorseIR::Type::Kind::Basic:
			DispatchBasic<G, N...>(generator, static_cast<const HorseIR::BasicType *>(type), nodes...);
			break;
		case HorseIR::Type::Kind::List:
			DispatchList<G, N...>(generator, static_cast<const HorseIR::ListType *>(type), nodes...);
			break;
		default:
			Utils::Logger::LogError("Unsupported type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' in function " + generator.m_builder.GetContextString("Dispatch"));
	}
}

template<class G, typename... N>
static void DispatchBasic(G &generator, const HorseIR::BasicType *type, N ...nodes)
{
	switch (type->GetBasicKind())
	{
		case HorseIR::BasicType::BasicKind::Boolean:
			generator.template Generate<PTX::PredicateType>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Char:
		case HorseIR::BasicType::BasicKind::Int8:
			generator.template Generate<PTX::Int8Type>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Int16:
			generator.template Generate<PTX::Int16Type>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Int32:
			generator.template Generate<PTX::Int32Type>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Int64:
			generator.template Generate<PTX::Int64Type>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Float32:
			generator.template Generate<PTX::Float32Type>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Float64:
			generator.template Generate<PTX::Float64Type>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Date:
			generator.template Generate<PTX::Int32Type>(nodes...);
			break;
		case HorseIR::BasicType::BasicKind::String:
		case HorseIR::BasicType::BasicKind::Symbol:
			generator.template Generate<PTX::UInt64Type>(nodes...);
			break;
		default:
			Utils::Logger::LogError("Unsupported type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' in function " + generator.m_builder.GetContextString("Dispatch"));
	}
}

template<class G, typename... N>
static void DispatchList(G &generator, const HorseIR::ListType *type, N ...nodes)
{
	if (const auto elementType = HorseIR::TypeUtils::GetSingleType(type->GetElementTypes()))
	{
		if (const auto basicType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(elementType))
		{
			DispatchBasic(generator, basicType, nodes...);
			return;
		}
	}
	Utils::Logger::LogError("Unsupported type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' in function " + generator.m_builder.GetContextString("Dispatch"));
}

}
