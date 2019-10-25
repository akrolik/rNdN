#pragma once

#include <cmath>
#include <utility>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"

#include "PTX/PTX.h"

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
		Global,
		CellData,
		Cell
	};

	std::string KindString(Kind kind)
	{
		switch (kind)
		{
			case Kind::Null:
				return "null";
			case Kind::Lane:
				return "lane";
			case Kind::Warp:
				return "warp";
			case Kind::Local:
				return "local";
			case Kind::Block:
				return "block";
			case Kind::Global:
				return "global";
			case Kind::CellData:
				return "celldata";
			case Kind::Cell:
				return "cell";
		}
		return "unknown";
	}

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
			case Kind::CellData:
				return GenerateCellDataIndex();
			case Kind::Cell:
				return GenerateCellIndex();
		}
		return nullptr;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateLaneIndex()
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

	const PTX::Register<PTX::UInt32Type> *GenerateWarpIndex()
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

	const PTX::Register<PTX::UInt32Type> *GenerateLocalIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srtidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_tid->GetVariable("%tid"), PTX::VectorElement::X);

		auto tidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(tidx, srtidx));

		return tidx;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateBlockIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);

		auto ctaidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ctaidx, srctaidx));

		return ctaidx;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateGlobalIndex()
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

		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		auto madInstruction = new PTX::MADInstruction<PTX::UInt32Type>(index, ctaidx, ntidx, tidx);
		madInstruction->SetLower(true);
		this->m_builder.AddStatement(madInstruction);

		return index;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateInitialCellDataIndex()
	{
		// Cell index = globalIndex % cellThreads

		auto resources = this->m_builder.GetLocalResources();

		auto index = resources->AllocateRegister<PTX::UInt32Type>(NameUtils::GeometryThreadIndex);

		auto globalIndex = GenerateGlobalIndex();
		auto cellThreads = GenerateCellThreads();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(index, globalIndex, cellThreads));

		return index;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateCellDataIndex()
	{
		auto resources = this->m_builder.GetLocalResources();

		// Fetch the special register which is computed for each thread and updated by the loop

		return resources->GetRegister<PTX::UInt32Type>(NameUtils::GeometryThreadIndex);
	}

	const PTX::Register<PTX::UInt32Type> *GenerateCellIndex()
	{
		// Cell index = globalIndex / cellThreads

		auto resources = this->m_builder.GetLocalResources();

		auto globalIndex = GenerateGlobalIndex();
		auto cellThreads = GenerateCellThreads();

		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(index, globalIndex, cellThreads));

		return index;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateCellThreads()
	{
		auto& inputOptions = m_builder.GetInputOptions();

		if (inputOptions.ListCellThreads == InputOptions::DynamicSize)
		{
			auto resources = this->m_builder.GetLocalResources();
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::GeometryCellThreads);
		}
		else
		{
			// If the thread count is specified, we can use a constant value

			return new PTX::UInt32Value(inputOptions.ListCellThreads);
		}
	}
};

}
