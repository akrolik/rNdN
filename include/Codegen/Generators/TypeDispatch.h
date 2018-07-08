#pragma once


// @method Dispatch<G> (G = Generator)
//
// @requires
// 	- G has a type field generator->NodeType
// 	- G is a class containing a templated static method
// 		template<class G>
// 		static void Generate(NodeType*, Builder*)
//
// Convenience method for converting between HorseIR dynamic types and PTX static types, and
// instantiating the statically typed PTX generator.
//
// This allows the code generator to centralize the type mapping for various constructs
// (assignments, returns, parameters) in a single place.

namespace Codegen {

template<class G>
static void Dispatch(G *generator, HorseIR::Type *type, const typename G::NodeType *node)
{
	switch (type->GetKind())
	{
		case HorseIR::Type::Kind::Primitive:
			Dispatch<G>(generator, static_cast<HorseIR::PrimitiveType *>(type), node);
			break;
		case HorseIR::Type::Kind::List:
			Dispatch<G>(generator, static_cast<HorseIR::ListType *>(type), node);
			break;
		default:
			//TODO:
			// std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetContextString(TO_STRING(G)) << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

template<class G>
static void Dispatch(G *generator, HorseIR::PrimitiveType *type, const typename G::NodeType *node)
{
	switch (type->GetKind())
	{
		case HorseIR::PrimitiveType::Kind::Bool:
			generator->template Generate<PTX::PredicateType>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int8:
			generator->template Generate<PTX::Int8Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int16:
			generator->template Generate<PTX::Int16Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int32:
			generator->template Generate<PTX::Int32Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Int64:
			generator->template Generate<PTX::Int64Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Float32:
			generator->template Generate<PTX::Float32Type>(node);
			break;
		case HorseIR::PrimitiveType::Kind::Float64:
			generator->template Generate<PTX::Float64Type>(node);
			break;
		default:
			//TODO:
			// std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in function " << m_builder->GetContextString(TO_STRING(G)) << std::endl;
			std::exit(EXIT_FAILURE);
	}
}

template<class G>
static void Dispatch(G *generator, HorseIR::ListType *type, const typename G::NodeType *node)
{
	Dispatch<G>(static_cast<HorseIR::PrimitiveType *>(type->GetElementType()), node);
	//TODO:
	// std::cerr << "[ERROR] Unsupported type " << type->ToString() << " in context " << m_builder->GetContextString(TO_STRING(G)) << std::endl;
	// std::exit(EXIT_FAILURE);
}

}
