#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Instructions/Comparison/SelectInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"

namespace Codegen {

template<PTX::Bits B>
class ReturnGenerator : public Generator
{
public:
	using Generator::Generator;

	using IndexKind = typename AddressGenerator<B>::IndexKind;

	void Generate(const HorseIR::ReturnStatement *returnS, IndexKind indexKind)
	{
		for (const auto& operand : returnS->GetOperands())
		{
			Codegen::DispatchType(*this, operand->GetType(), operand, indexKind);
		}
	}

	template<class T>
	void Generate(const HorseIR::Operand *operand, IndexKind indexKind)
	{
		// Check if the kernel is returned by one of the generators using an atomic action

		//TODO: Atomic return may be only a single variable
		auto& kernelOptions = this->m_builder.GetKernelOptions();
		if (kernelOptions.IsAtomicReturn())
		{
			return;
		}

		// Otherwise we use a typical store return

		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate (1-bit) values are stored as 8 bit integers on the CPU side
			// so a conversion must first be run

			OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(operand);
			auto converted = ConversionGenerator::ConvertSource<PTX::Int8Type>(this->m_builder, value);

			Generate(converted, indexKind);
		}
		else
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(operand);
			Generate(value, indexKind);
		}
	}

	template<class T, typename Enable = std::enable_if_t<PTX::StoreInstruction<B, T, PTX::GlobalSpace, PTX::StoreSynchronization::Weak, false>::TypeSupported>>
	void Generate(const PTX::Register<T> *value, IndexKind indexKind)
	{
		// Fetch the return variable

		auto kernelResources = this->m_builder.GetKernelResources();
		auto variable = kernelResources->template GetParameter<PTX::PointerType<B, T>, PTX::ParameterSpace>("$return");

		// Store the value at the appropriate index

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(variable, indexKind);

		this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}
};

}
