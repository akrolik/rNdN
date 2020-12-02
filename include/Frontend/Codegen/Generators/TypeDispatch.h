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

namespace Frontend {
namespace Codegen {

template<typename T>
class VectorDispatch
{
public:
	template<class G, typename... N>
	static void Dispatch(G &generator, unsigned int i, N ...nodes)
	{
		generator.template GenerateVector<T>(nodes...);
	}
};

template<typename T>
class ListHomogeneousDispatch
{
public:
	template<class G, typename... N>
	static void Dispatch(G &generator, unsigned int i, N ...nodes)
	{
		generator.template GenerateList<T>(nodes...);
	}
};

template<typename T>
class ListHeterogeneousDispatch
{
public:
	template<class G, typename... N>
	static void Dispatch(G &generator, unsigned int i, N ...nodes)
	{
		generator.template GenerateTuple<T>(i, nodes...);
	}
};

template<template <typename> typename D, class G, typename... N>
static void Dispatch(G &generator, const HorseIR::BasicType *type, unsigned int i, N ...nodes)
{
	switch (type->GetBasicKind())
	{
		case HorseIR::BasicType::BasicKind::Boolean:
			D<PTX::PredicateType>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Char:
		case HorseIR::BasicType::BasicKind::Int8:
			D<PTX::Int8Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Int16:
			D<PTX::Int16Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Int32:
			D<PTX::Int32Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Int64:
			D<PTX::Int64Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Float32:
			D<PTX::Float32Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Float64:
			D<PTX::Float64Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::String:
		case HorseIR::BasicType::BasicKind::Symbol:
			D<PTX::UInt64Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Date:
		case HorseIR::BasicType::BasicKind::Month:
		case HorseIR::BasicType::BasicKind::Minute:
		case HorseIR::BasicType::BasicKind::Second:
			D<PTX::Int32Type>::Dispatch(generator, i, nodes...);
			break;
		case HorseIR::BasicType::BasicKind::Datetime:
		case HorseIR::BasicType::BasicKind::Time:
			D<PTX::Int64Type>::Dispatch(generator, i, nodes...);
			break;
		default:
			Utils::Logger::LogError("Unsupported type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' in function " + generator.m_builder.GetContextString("Dispatch"));
	}
}

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
	Dispatch<VectorDispatch, G, N...>(generator, type, 0, nodes...);
}

template<class G, typename... N>
static void DispatchList(G &generator, const HorseIR::ListType *type, N ...nodes)
{
	if (const auto elementType = HorseIR::TypeUtils::GetSingleType(type->GetElementTypes()))
	{
		if (const auto basicType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(elementType))
		{
			Dispatch<ListHomogeneousDispatch, G, N...>(generator, basicType, 0, nodes...);
		}
		else
		{
			Utils::Logger::LogError("Unsupported type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' in function " + generator.m_builder.GetContextString("Dispatch"));
		}
	}
	else
	{
		const auto& elementTypes = type->GetElementTypes();
		for (auto i = 0u; i < elementTypes.size(); ++i)
		{
			const auto elementType = elementTypes.at(i);
			if (const auto basicType = HorseIR::TypeUtils::GetType<HorseIR::BasicType>(elementType))
			{
				Dispatch<ListHeterogeneousDispatch, G, N...>(generator, basicType, i, nodes...);
			}
			else
			{
				Utils::Logger::LogError("Unsupported type '" + HorseIR::PrettyPrinter::PrettyString(type) + "' in function " + generator.m_builder.GetContextString("Dispatch"));
			}
		}
	}
}

}
}
