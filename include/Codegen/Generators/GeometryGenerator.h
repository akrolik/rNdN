#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/IndexGenerator.h"

#include "PTX/PTX.h"

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
		auto resources = this->m_builder.GetLocalResources();

		// We cannot operate directly on special registers, so they must first be copied to a user defined register

		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);

		auto ntidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto count = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));
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

		auto srntidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ntid->GetVariable("%ntid"), PTX::VectorElement::X);
		auto srctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_ctaid->GetVariable("%ctaid"), PTX::VectorElement::X);
		auto srnctaidx = new PTX::IndexedRegister4<PTX::UInt32Type>(PTX::SpecialRegisterDeclaration_nctaid->GetVariable("%nctaid"), PTX::VectorElement::X);

		auto ntidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto ctaidx = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto nctaidx = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ntidx, srntidx));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(ctaidx, srctaidx));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(nctaidx, srnctaidx));

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
