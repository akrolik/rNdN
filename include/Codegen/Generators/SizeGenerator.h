#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class SizeGenerator : public Generator
{
public:
	using Generator::Generator;
	
	const PTX::TypedOperand<PTX::UInt64Type> *GenerateInputSize()
	{
		auto& inputOptions = m_builder.GetInputOptions();

		if (inputOptions.ActiveThreads == InputOptions::DynamicSize)
		{
			//TODO: Check bounds
			Utils::Logger::LogError("Non-constant number of threads");
		}
		else
		{
			// If the input size is specified, we can use a constant value

			return new PTX::UInt64Value(inputOptions.ActiveThreads);
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
