#pragma once

#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Tree.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "Utils/Logger.h"

namespace Codegen {

class OperandCompressionGenerator : public HorseIR::ConstVisitor, public Generator
{
public:
	using Generator::Generator;

	static const PTX::Register<PTX::PredicateType> *UnaryCompressionRegister(Builder& builder, const std::vector<HorseIR::Operand *>& arguments) 
	{
		// Propagate the compression mask used as input

		OperandCompressionGenerator compGen(builder);
		return compGen.GetCompressionRegister(arguments.at(0));
	}

	static const PTX::Register<PTX::PredicateType> *BinaryCompressionRegister(Builder& builder, const std::vector<HorseIR::Operand *>& arguments)
	{
		// Ensure either both paths are compressed with the same mask, or that we only mask on one path
		// (and therefore are shrinking one variable to use with another)

		OperandCompressionGenerator compGen(builder);
		auto compression1 = compGen.GetCompressionRegister(arguments.at(0));
		auto compression2 = compGen.GetCompressionRegister(arguments.at(1));

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
		Utils::Logger::LogError("Compression registers differ and are non-null");
	}

	const PTX::Register<PTX::PredicateType> *GetCompressionRegister(const HorseIR::Expression *expression)
	{
		m_compression = nullptr;
		expression->Accept(*this);
		return m_compression;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class S>
	void Generate(const HorseIR::Identifier *identifier)
	{
		//GLOBAL: Support identifiers which contain a module name

		// Only fetch compression for registers which already exist (this handles non-reassigned parameters which do ont have compression)

		auto resources = this->m_builder.GetLocalResources();
		if (resources->ContainsRegister<S>(identifier->GetName()))
		{
			m_compression = resources->GetCompressionRegister<S>(identifier->GetName());
		}
	}

private:
	const PTX::Register<PTX::PredicateType> *m_compression = nullptr;
};

}
