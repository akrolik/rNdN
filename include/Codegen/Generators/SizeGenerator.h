#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "PTX/Type.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Instructions/Arithmetic/DivideInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Operands/SpecialConstants.h"
#include "PTX/Operands/Variables/IndexedRegister.h"

namespace Codegen {

template<PTX::Bits B>
class SizeGenerator : public Generator
{
public:
	using Generator::Generator;
	
	const PTX::TypedOperand<PTX::UInt64Type> *GenerateInputSize()
	{
		auto inputOptions = m_builder.GetInputOptions();

		if (inputOptions.InputSize == InputOptions::DynamicSize)
		{
			// If the input size is dyanmic, then it will be passed as a parameter
			// Load it from the parameter space

			auto resources = this->m_builder.GetLocalResources();
			auto functionResources = this->m_builder.GetFunctionResources();

			// Get the size variable from the function resources

			auto sizeVariable = functionResources->template GetParameter<PTX::UInt64Type, PTX::ParameterSpace>("$size");

			auto sizeAddress = new PTX::MemoryAddress<B, PTX::UInt64Type, PTX::ParameterSpace>(sizeVariable);
			auto size = resources->template AllocateTemporary<PTX::UInt64Type>("size");
			this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::UInt64Type, PTX::ParameterSpace>(size, sizeAddress));

			return size;
		}
		else
		{
			// If the input size is specified, we can use a constant value

			return new PTX::UInt64Value(inputOptions.InputSize);
		}
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateWarpCount()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);

		auto ntidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto count = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));
		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(count, ntidx, PTX::SpecialConstant_WARP_SZ));

		return count;
	}
};

}
