#pragma once

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/SpecialRegisterGenerator.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

class GeometryGenerator : public Generator
{
public:
	using Generator::Generator;

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateVectorSize()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			// If the input size is specified, we can use a constant value. Otherwise, use the special parameter

			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(vectorShape->GetSize()))
			{
				return new PTX::UInt32Value(constantSize->GetValue());
			}

			auto resources = this->m_builder.GetLocalResources();
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::GeometryDataSize);
		}
		
		Utils::Logger::LogError("Unknown vector size for thread geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateListSize()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			// If the input size is specified, we can use a constant value. Otherwise, use the special parameter

			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				return new PTX::UInt32Value(constantSize->GetValue());
			}

			auto resources = this->m_builder.GetLocalResources();
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::GeometryListSize);
		}
		
		Utils::Logger::LogError("Unknown list size for thread geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateCellSize()
	{
		//TODO: Specify the cell sizes as part of the kernel code
		auto resources = this->m_builder.GetLocalResources();
		return resources->GetRegister<PTX::UInt32Type>(NameUtils::GeometryDataSize);
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateWarpCount()
	{
		// Warp count = ntidx / warp size

		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto ntidx = specialGenerator.GenerateThreadCount();

		auto resources = this->m_builder.GetLocalResources();
		auto count = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(count, ntidx, PTX::SpecialConstant_WARP_SZ));

		return count;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateActiveThreads()
	{
		auto resources = this->m_builder.GetLocalResources();

		// Compute the number of active threads in a block for list geometry
		//   - Blocks 0...(N-1): ntid
		//   - Block N: (CellCount * CellThreads) % ntidx
		// We then select based on the ctaid

		IndexGenerator indexGenerator(this->m_builder);
		auto cellThreads = indexGenerator.GenerateCellThreads();
		auto cellCount = GenerateListSize();

		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto ntidx = specialGenerator.GenerateThreadCount();
		auto ctaidx = specialGenerator.GenerateBlockIndex();
		auto nctaidx = specialGenerator.GenerateBlockCount();

		auto threads = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto roundThreads = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto activeThreads = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto temp = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		auto multiply = new PTX::MultiplyInstruction<PTX::UInt32Type>(threads, cellCount, cellThreads);
		multiply->SetLower(true);
		this->m_builder.AddStatement(multiply);

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(roundThreads, threads, ntidx));
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(temp, ctaidx, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, temp, nctaidx, PTX::UInt32Type::ComparisonOperator::Equal));
		this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(activeThreads, roundThreads, ntidx, predicate));

		return activeThreads;
	}
};

}
