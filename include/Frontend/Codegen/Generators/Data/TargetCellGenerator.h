#pragma once

#include "Frontend/Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/NameUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B, class T>
class TargetCellGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "TargetCellGenerator"; }

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, unsigned int index)
	{
		m_index = index;
		target->Accept(*this);
		return m_targetRegister;
	}

	void Visit(const HorseIR::VariableDeclaration *declaration) override
	{
		auto resources = this->m_builder.GetLocalResources();
		auto name = NameUtils::VariableName(declaration, m_index);

		m_targetRegister = resources->template AllocateRegister<T>(name);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		auto resources = this->m_builder.GetLocalResources();
		auto name = NameUtils::VariableName(identifier, m_index);

		m_targetRegister = resources->template AllocateRegister<T>(name);
	}

private:
	unsigned int m_index = 0;
	const PTX::Register<T> *m_targetRegister = nullptr;
};

}
}
