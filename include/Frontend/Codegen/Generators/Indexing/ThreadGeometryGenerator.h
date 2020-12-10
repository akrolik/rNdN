#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/NameUtils.h"
#include "Frontend/Codegen/Generators/Indexing/SpecialRegisterGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class ThreadGeometryGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ThreadGeometryGenerator"; }

	PTX::TypedOperand<PTX::UInt32Type> *GenerateVectorGeometry()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			// If the input size is specified, we can use a constant value. Otherwise, use the special parameter

			if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(vectorGeometry->GetSize()))
			{
				return new PTX::UInt32Value(constantSize->GetValue());
			}

			auto resources = this->m_builder.GetLocalResources();
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryDataSize);
		}
		Error("vector size for thread geometry");
	}

	PTX::TypedOperand<PTX::UInt32Type> *GenerateListGeometry()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto listGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			// If the input size is specified, we can use a constant value. Otherwise, use the special parameter

			if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listGeometry->GetListSize()))
			{
				return new PTX::UInt32Value(constantSize->GetValue());
			}

			auto resources = this->m_builder.GetLocalResources();
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryListSize);
		}
		Error("list size for thread geometry");
	}

	PTX::TypedOperand<PTX::UInt32Type> *GenerateListDataGeometry()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto listGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			// If the input size is specified and the same for all cells, we can use a constant value. Otherwise, use the special parameter

			auto cellGeometry = HorseIR::Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
			if (const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(cellGeometry))
			{
				if (const auto constantSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(vectorGeometry->GetSize()))
				{
					return new PTX::UInt32Value(constantSize->GetValue());
				}
			}

			auto resources = this->m_builder.GetLocalResources();
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryDataSize);
		}
		Error("cell size for thread geometry");
	}

	PTX::TypedOperand<PTX::UInt32Type> *GenerateDataGeometry()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (inputOptions.IsVectorGeometry())
		{
			return GenerateVectorGeometry();
		}
		else if (inputOptions.IsListGeometry())
		{
			return GenerateListDataGeometry();
		}
		Error("data size for thread geometry");
	}

	PTX::TypedOperand<PTX::UInt32Type> *GenerateWarpCount()
	{
		// Warp count = ntidx / warp size

		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto ntidx = specialGenerator.GenerateThreadCount();

		auto resources = this->m_builder.GetLocalResources();
		auto count = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(count, ntidx, PTX::SpecialConstant_WARP_SZ));

		return count;
	}

	PTX::TypedOperand<PTX::UInt32Type> *GenerateListThreads()
	{
		// Check if the thread count is specified as part of the input options or is dynamic

		auto& inputOptions = m_builder.GetInputOptions();
		if (inputOptions.ListCellThreads == InputOptions::DynamicSize)
		{
			auto resources = this->m_builder.GetLocalResources();
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryListThreads);
		}
		else
		{
			// If the thread count is specified, we can use a constant value

			return new PTX::UInt32Value(inputOptions.ListCellThreads);
		}
	}

	PTX::TypedOperand<PTX::UInt32Type> *GenerateActiveThreads()
	{
		auto resources = this->m_builder.GetLocalResources();

		// Compute the number of active threads in a block for list geometry
		//   - Blocks 0...(N-1): ntid
		//   - Block N: (ListSize * ListThreads) % ntidx
		// We then select based on the ctaid

		auto listThreads = GenerateListThreads();
		auto listSize = GenerateListGeometry();

		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto ntidx = specialGenerator.GenerateThreadCount();
		auto ctaidx = specialGenerator.GenerateBlockIndex();
		auto nctaidx = specialGenerator.GenerateBlockCount();

		auto threads = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto roundThreads = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto activeThreads = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto temp = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::MultiplyInstruction<PTX::UInt32Type>(
			threads, listSize, listThreads, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
		));
		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(roundThreads, threads, ntidx));
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(temp, ctaidx, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			predicate, temp, nctaidx, PTX::UInt32Type::ComparisonOperator::Equal
		));
		this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(activeThreads, roundThreads, ntidx, predicate));

		return activeThreads;
	}
};

}
}
