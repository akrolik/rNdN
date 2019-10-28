#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class TargetGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	const PTX::Register<T> *Generate(const HorseIR::LValue *target, const PTX::Register<PTX::PredicateType> *predicateRegister)
	{
		target->Accept(*this);

		// Compress the target register if needed

		if (predicateRegister != nullptr)
		{
			auto resources = this->m_builder.GetLocalResources();
			resources->SetCompressedRegister<T>(m_targetRegister, predicateRegister);
		}
		return m_targetRegister;
	}

	void Visit(const HorseIR::VariableDeclaration *declaration) override
	{
		// For declarations, allocate a new register

		auto resources = this->m_builder.GetLocalResources();
		auto name = NameUtils::VariableName(declaration);

		m_targetRegister = resources->template AllocateRegister<T>(name);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		auto resources = this->m_builder.GetLocalResources();

		// Check if the identifier has already been allocated a register

		auto name = NameUtils::VariableName(identifier);
		if (resources->ContainsRegister<T>(name))
		{
			// If the variable has already been defined, we either compress the output, or leave as is

			m_targetRegister = resources->GetRegister<T>(name);
		}
		else
		{
			// In the case that it has not, then it *must* be a parameter - all local declarations
			// must already have a register

			auto& parameterShapes = this->m_builder.GetInputOptions().ParameterShapes;
			if (parameterShapes.find(identifier->GetSymbol()) != parameterShapes.end())
			{
				// For parameters, we then allocate - this allows us to distinguish between the initial
				// and re-assigned values for parameters

				m_targetRegister = resources->template AllocateRegister<T>(name);
			}
			else
			{
				Utils::Logger::LogError("Unable to find register for target '" + HorseIR::PrettyPrinter::PrettyString(identifier) + "'");
			}
		}
	}

private:
	const PTX::Register<T> *m_targetRegister = nullptr;
};

}
