#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"

#include "HorseIR/Tree/Parameter.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Statements/BlockStatement.h"

namespace Codegen {

template<PTX::Bits B>
class ParameterGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	void Generate(const HorseIR::Parameter *parameter)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto declaration = new PTX::PointerDeclaration<B, PTX::Int8Type>(parameter->GetName());
			this->m_builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(parameter->GetName());
			auto value = this->m_builder->AllocateRegister<T>(parameter->GetName());

			auto block = new PTX::BlockStatement();
			this->m_builder->AddStatement(block);
			this->m_builder->OpenScope(block);

			auto temp8 = this->m_builder->AllocateRegister<PTX::Int8Type, ResourceKind::Internal>(parameter->GetName());
			auto temp16 = this->m_builder->AllocateRegister<PTX::Int16Type, ResourceKind::Internal>(parameter->GetName());

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto address = addressGenerator.template Generate<PTX::Int8Type>(variable);

			this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::Int8Type, PTX::GlobalSpace>(temp8, address));
			this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int16Type, PTX::Int8Type>(temp16, temp8));
			this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(value, temp16, new PTX::Value<PTX::Int16Type>(0), PTX::Int16Type::ComparisonOperator::NotEqual));

			this->m_builder->CloseScope();
		}
		else
		{
			auto declaration = new PTX::PointerDeclaration<B, T>(parameter->GetName());
			this->m_builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(parameter->GetName());
			auto value = this->m_builder->AllocateRegister<T>(parameter->GetName());

			auto block = new PTX::BlockStatement();
			this->m_builder->AddStatement(block);
			this->m_builder->OpenScope(block);

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto address = addressGenerator.template Generate<T>(variable);

			this->m_builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(value, address));

			this->m_builder->CloseScope();
		}
	}
};

}
