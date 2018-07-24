#pragma once

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Expressions/Identifier.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"

namespace Codegen {

class OperandCompressionGenerator : public HorseIR::ForwardTraversal, public Generator
{
public:
	using Generator::Generator;

	static const PTX::Register<PTX::PredicateType> *UnaryCompressionRegister(Builder *builder, const HorseIR::CallExpression *call) 
	{
		OperandCompressionGenerator compGen(builder);
		return compGen.GetCompressionRegister(call->GetArgument(0));
	}

	static const PTX::Register<PTX::PredicateType> *BinaryCompressionRegister(Builder *builder, const HorseIR::CallExpression *call)
	{
		OperandCompressionGenerator compGen(builder);
		auto compression1 = compGen.GetCompressionRegister(call->GetArgument(0));
		auto compression2 = compGen.GetCompressionRegister(call->GetArgument(1));

		if (compression1 == compression2)
		{
			return compression1;
		}
		else if (compression1 == nullptr)
		{
			return compression2;
		}
		else if (compression2 == nullptr)
		{
			return compression1;
		}
		std::cerr << "[ERROR] Compression registers differ and are non-null" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	const PTX::Register<PTX::PredicateType> *GetCompressionRegister(HorseIR::Expression *expression)
	{
		m_compression = nullptr;
		expression->Accept(*this);
		return m_compression;
	}

	void Visit(HorseIR::Identifier *identifier) override
	{
		Codegen::DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class S>
	void Generate(const HorseIR::Identifier *identifier)
	{
		m_compression = this->m_builder->GetCompressionRegister<S>(identifier->GetString());
	}

private:
	const PTX::Register<PTX::PredicateType> *m_compression = nullptr;
};

}
