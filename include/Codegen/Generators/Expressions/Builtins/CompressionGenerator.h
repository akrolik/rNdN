#pragma once

#include "HorseIR/Traversal/ConstForwardTraversal.h"

#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"

#include "PTX/Statements/CommentStatement.h"
#include "PTX/Instructions/Logical/AndInstruction.h"

namespace Codegen {

template<PTX::Bits B, class T>
class CompressionGenerator : public BuiltinGenerator<B, T>, public HorseIR::ConstForwardTraversal
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	void Generate(const std::string& target, const HorseIR::CallExpression *call) override
	{
		// Update the resource generator with the compression information. @compress arguments:
		//   0 - bool mask
		//   1 - value

		OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
		m_predicate = opGen.GenerateRegister(call->GetArgument(0));
		m_target = target;

		call->GetArgument(1)->Accept(*this);
	}

	void Visit(const HorseIR::Expression *expression) override
	{
		BuiltinGenerator<B, T>::Unimplemented("compression of non-identifier kind");
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		// Compression does not create a new register, instead it creates a mapping
		// from an identifier to a register-predicate pair
		// 
		// i.e. Given a compression call
		//
		//         t1:i32 = @compress(p, t0);
		//
		//      the mapping t1 -> (t0, p) is created in the resource allocator. Future
		//      lookups for the identifier t1 will produce register t0

		OperandGenerator<B, T> opGen(this->m_builder);
		auto value = opGen.GenerateRegister(identifier);

		auto resources = this->m_builder.GetLocalResources();
		auto compression = resources->template GetCompressionRegister<T>(identifier->GetString());

		if (compression != nullptr)
		{
			// If a predicate has already been set on the value register, combine
			// it with the new compress predicate

			auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(predicate, compression, m_predicate));
			resources->AddCompressedRegister(m_target, value, predicate);
		}
		else
		{
			// Otherwise this is the first compression and it is simply stored

			resources->AddCompressedRegister(m_target, value, m_predicate);
		}
	}

private:
	std::string m_target;
	const PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
};

}
