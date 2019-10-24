#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class IndexGenerator : public HorseIR::ConstVisitor, public Generator
{
public:
	using Generator::Generator;

	const PTX::Register<T> *GenerateOperand(const HorseIR::Expression *expression, const PTX::TypedOperand<PTX::UInt32Type> *index, const std::string& destination)
	{
		m_index = index;
		m_destination = destination;

		m_register = nullptr;
		expression->Accept(*this);
		if (m_register != nullptr)
		{
			return m_register;
		}

		Utils::Logger::LogError("Unable to generate operand '" + HorseIR::PrettyPrinter::PrettyString(expression) + "' with index " + index->ToString());
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class S>
	void Generate(const HorseIR::Identifier *identifier)
	{
		// Determine if the identifier is a local variable or parameter

		auto& parameterShapes = this->m_builder.GetInputOptions().ParameterShapes;
		if (parameterShapes.find(identifier->GetSymbol()) != parameterShapes.end())
		{
			//GLOBAL: Identifiers may be global and have a module name
			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			auto source = loadGenerator.template GeneratePointer<S>(identifier->GetName(), m_index, m_destination);
			if constexpr(std::is_same<T, S>::value)
			{
				m_register = source;
			}
			else
			{
				m_register = ConversionGenerator::ConvertSource<T, S>(this->m_builder, source);
			}
		}
	}

private:
	const PTX::Register<T> *m_register = nullptr;

	const PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
	const std::string m_destination;
};

}

