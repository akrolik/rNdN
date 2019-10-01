#pragma once

#include <cmath>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "PTX/Type.h"
#include "PTX/Declarations/SpecialRegisterDeclarations.h"
#include "PTX/Instructions/Arithmetic/DivideInstruction.h"
#include "PTX/Instructions/Arithmetic/MADInstruction.h"
#include "PTX/Instructions/Arithmetic/RemainderInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Operands/SpecialConstants.h"
#include "PTX/Operands/Value.h"
#include "PTX/Operands/Variables/IndexedRegister.h"

namespace Codegen {

class IndexGenerator : public Generator
{
public:
	using Generator::Generator;

	enum class Kind {
		Null,
		Lane,
		Warp,
		Local,
		Block,
		Global
	};

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateIndex(Kind kind)
	{
		switch (kind)
		{
			case Kind::Null:
				return new PTX::UInt32Value(0);
			case Kind::Lane:
				return GenerateLaneIndex();
			case Kind::Warp:
				return GenerateWarpIndex();
			case Kind::Local:
				return GenerateLocalIndex();
			case Kind::Block:
				return GenerateBlockIndex();
			case Kind::Global:
				return GenerateGlobalIndex();
		}
		return nullptr;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateLaneIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);

		auto tidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto laneid = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(laneid, tidx, PTX::SpecialConstant_WARP_SZ));

		return laneid;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateWarpIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);

		auto tidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto warpid = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));
		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(warpid, tidx, PTX::SpecialConstant_WARP_SZ));

		return warpid;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateLocalIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);

		auto tidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));

		return tidx;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateBlockIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);

		auto ctaidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ctaidx, srctaidx));

		return ctaidx;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateGlobalIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);
		auto srctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);
		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);

		auto tidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto ctaidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto ntidx = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ctaidx, srctaidx));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));

		// Compute the thread index as blockSize * blockIndex + threadIndex

		auto index = resources->template AllocateTemporary<PTX::UInt32Type>("index");

		auto madInstruction = new PTX::MADInstruction<PTX::UInt32Type>(index, ctaidx, ntidx, tidx);
		madInstruction->SetLower(true);
		this->m_builder.AddStatement(madInstruction);

		return index;
	}
};

}
