#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Expressions/Builtins/BuiltinGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/OperandCompressionGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class CompressionGenerator : public BuiltinGenerator<B, T>, public HorseIR::ConstVisitor
{
public:
	using BuiltinGenerator<B, T>::BuiltinGenerator;

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments) override
	{
		// Update the resource generator with the compression information. @compress arguments:
		//   0 - bool mask
		//   1 - value

		OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
		m_predicate = opGen.GenerateRegister(arguments.at(0), OperandGenerator<B, PTX::PredicateType>::LoadKind::Vector);
		m_target = target;

		arguments.at(1)->Accept(*this);

		return m_targetRegister;
	}

	void Visit(const HorseIR::Expression *expression) override
	{
		BuiltinGenerator<B, T>::Unimplemented("compression of non-identifier kind");
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		auto resources = this->m_builder.GetLocalResources();

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
		m_targetRegister = opGen.GenerateRegister(identifier, OperandGenerator<B, T>::LoadKind::Vector);

		OperandCompressionGenerator compGen(this->m_builder);
		auto compression = compGen.GetCompressionRegister(identifier);

		if (compression != nullptr)
		{
			// If a predicate has already been set on the value register, combine
			// it with the new compress predicate

			auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(predicate, compression, m_predicate));
			//GLOBAL: Global variables have a module name
			resources->AddCompressedRegister(m_target->GetSymbol()->name, m_targetRegister, predicate);
		}
		else
		{
			// Otherwise this is the first compression and it is simply stored

			//GLOBAL: Global variables have a module name
			resources->AddCompressedRegister(m_target->GetSymbol()->name, m_targetRegister, m_predicate);
		}
	}

private:
	const HorseIR::LValue *m_target = nullptr;
	const PTX::Register<T> *m_targetRegister = nullptr;
	const PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
};

}
