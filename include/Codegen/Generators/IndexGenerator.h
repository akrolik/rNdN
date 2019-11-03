#pragma once

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/SpecialRegisterGenerator.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

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
		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto tidx = specialGenerator.GenerateThreadIndex();

		auto resources = this->m_builder.GetLocalResources();
		auto laneid = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(laneid, tidx, PTX::SpecialConstant_WARP_SZ));

		return laneid;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateWarpIndex()
	{
		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto tidx = specialGenerator.GenerateThreadIndex();

		auto resources = this->m_builder.GetLocalResources();
		auto warpid = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(warpid, tidx, PTX::SpecialConstant_WARP_SZ));

		return warpid;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateLocalIndex()
	{
		SpecialRegisterGenerator specialGenerator(this->m_builder);
		return specialGenerator.GenerateThreadIndex();
	}

	const PTX::Register<PTX::UInt32Type> *GenerateBlockIndex()
	{
		// Check if we use in order blocks (allocation happens at the beginning of execution), or using the system value

		auto& inputOptions = this->m_builder.GetInputOptions();
		if (inputOptions.InOrderBlocks)
		{
			// Load the special block index from shard memory

			auto kernelResources = this->m_builder.GetKernelResources();
			auto s_blockIndex = kernelResources->template GetSharedVariable<PTX::UInt32Type>("blockIndex");

			AddressGenerator<PTX::Bits::Bits64> addressGenerator(this->m_builder);
			auto s_blockIndexAddress = addressGenerator.GenerateAddress(s_blockIndex);

			auto resources = this->m_builder.GetLocalResources();
			auto blockIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::LoadInstruction<PTX::Bits::Bits64, PTX::UInt32Type, PTX::SharedSpace>(blockIndex, s_blockIndexAddress));

			return blockIndex;
		}
		else
		{
			SpecialRegisterGenerator specialGenerator(this->m_builder);
			return specialGenerator.GenerateBlockIndex();
		}
	}

	const PTX::Register<PTX::UInt32Type> *GenerateGlobalIndex()
	{
		// Compute the thread index as blockSize * blockIndex + threadIndex

		auto tidx = GenerateLocalIndex();
		auto ctaidx = GenerateBlockIndex();

		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto ntidx = specialGenerator.GenerateThreadCount();

		auto resources = this->m_builder.GetLocalResources();
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

		auto globalIndex = GenerateGlobalIndex();
		auto cellThreads = GenerateCellThreads();

		auto resources = this->m_builder.GetLocalResources();
		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(index, globalIndex, cellThreads));

		return index;
	}

	const PTX::Register<PTX::UInt32Type> *GenerateDataIndex()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			return GenerateGlobalIndex();
		}
		else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			return GenerateCellDataIndex();
		}
		else
		{
			Utils::Logger::LogError("Unable to generate data index for geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
		}
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateCellThreads()
	{
		// Check if the thread count is specified as part of the input options or is dynamic

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
