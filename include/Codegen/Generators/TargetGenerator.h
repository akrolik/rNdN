#pragma once

#include "Codegen/Generators/Generator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B, class T>
class TargetGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const PTX::Register<PTX::PredicateType> *predicateRegister)
	{
		m_predicateRegister = predicateRegister;
		target->Accept(*this);
		return m_targetRegister;
	}

	void Visit(const HorseIR::VariableDeclaration *declaration) override
	{
		auto resources = this->m_builder.GetLocalResources();
		m_targetRegister = resources->template AllocateRegister<T>(declaration->GetName(), m_predicateRegister);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		auto resources = this->m_builder.GetLocalResources();
		if (m_predicateRegister == nullptr)
		{
			m_targetRegister = resources->GetRegister<T>(identifier->GetName());
		}
		else
		{
			m_targetRegister = resources->CompressRegister<T>(identifier->GetName(), m_predicateRegister);
		}
	}

private:
	const PTX::Register<PTX::PredicateType> *m_predicateRegister = nullptr;
	const PTX::Register<T> *m_targetRegister = nullptr;
};

}