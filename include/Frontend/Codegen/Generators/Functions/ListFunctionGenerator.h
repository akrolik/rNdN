#pragma once

#include "Frontend/Codegen/Generators/Functions/FunctionGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/NameUtils.h"
#include "Frontend/Codegen/Generators/Data/ParameterGenerator.h"
#include "Frontend/Codegen/Generators/Data/ParameterLoadGenerator.h"
#include "Frontend/Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadGeometryGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class ListFunctionGenerator : public FunctionGenerator<B>
{
public:
	using FunctionGenerator<B>::FunctionGenerator;

	std::string Name() const override { return "ListFunctionGenerator"; }

	void Generate(const HorseIR::Function *function)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto& inputOptions = this->m_builder.GetInputOptions();

		this->m_builder.AddStatement(new PTX::LocationDirective(this->m_builder.GetCurrentFile(), 0));

		// Setup the parameter (in/out) declarations in the kernel

		ParameterGenerator<B> parameterGenerator(this->m_builder);
		parameterGenerator.Generate(function);

		ParameterLoadGenerator<B> parameterLoadGenerator(this->m_builder);

		// Setup the thread count for each cell

		if (inputOptions.ListCellThreads == InputOptions::DynamicSize)
		{
			// Load the dynamic thread allocation parameter

			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::ThreadGeometryListThreads));

			auto parameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::ThreadGeometryListThreads);

			ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
			valueLoadGenerator.GenerateConstant(parameter);
		}
		else
		{
			// Restrict the number of threads in the cell block if we have already specified a size

			this->m_builder.SetBlockSize(inputOptions.ListCellThreads);
		}

		// Load the number of cells from the input if required

		const auto listGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(inputOptions.ThreadGeometry);
		if (HorseIR::Analysis::ShapeUtils::IsDynamicSize(listGeometry->GetListSize()))
		{
			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::ThreadGeometryListSize));

			auto parameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::ThreadGeometryListSize);

			ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
			valueLoadGenerator.GenerateConstant(parameter);
		}

		// Check of the cell is within bounds

		this->m_builder.AddStatement(new PTX::CommentStatement("bounds check"));

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto listIndex = indexGenerator.GenerateListIndex();

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto listSize = geometryGenerator.GenerateListGeometry();

		this->m_builder.AddIfStatement("LIST_BOUND", [&]()
		{
			auto exitPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				exitPredicate, listIndex, listSize, PTX::UInt32Type::ComparisonOperator::GreaterEqual
			));
			return std::make_tuple(exitPredicate, false);
		},
		[&]()
		{
			// Load the geometry cell sizes

			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::ThreadGeometryDataSize));

			auto geometryDataSizeParameter = parameterGenerator.template GeneratePointer<PTX::UInt32Type>(NameUtils::ThreadGeometryDataSize);
			parameterLoadGenerator.template GenerateParameterAddress<PTX::UInt32Type>(geometryDataSizeParameter);

			ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
			valueLoadGenerator.GeneratePointer(geometryDataSizeParameter, listIndex);

			// Initialize the special local index register

			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::ThreadGeometryListDataIndex));

			auto index = indexGenerator.GenerateListDataIndex();

			// Fetch the arguments from the .param address space

			parameterLoadGenerator.Generate(function);

			// Generate the starting index for the loop

			this->m_builder.AddStatement(new PTX::CommentStatement("loop init"));

			// Generate the loop bound

			auto listDataSize = geometryGenerator.GenerateListDataGeometry();
			auto bound = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(bound, index, listDataSize));

			// Construct the standard loop structure
			//
			//   <init>
			//   setp %p1, ...
			//   @%p1 br END
			//
			//   START:
			//      <body>
			//      <post>
			//
			//      setp %p2, ...
			//      @%p2 br START
			//
			//   END:

			this->m_builder.AddIfStatement("LIST_SKIP", [&]()
			{
				auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					predicate, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual
				));
				return std::make_tuple(predicate, false);
			},
			[&]()
			{
				this->m_builder.AddDoWhileLoop("LIST", [&](Builder::LoopContext& loopContext)
				{
					// Construct the loop body according to the standard function generator

					FunctionGenerator<B>::Visit(function);

					// Increment the thread index by the number of threads per cell

					this->m_builder.AddStatement(new PTX::CommentStatement("loop post"));

					auto listThreads = geometryGenerator.GenerateListThreads();
					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, listThreads));

					// Complete the loop structure

					auto predicateEnd = resources->template AllocateTemporary<PTX::PredicateType>();
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
						predicateEnd, index, bound, PTX::UInt32Type::ComparisonOperator::Less
					));
					return std::make_tuple(predicateEnd, false);
				});
			});
		});

		// Lastly, end with a return instruction - this only generated at the end for body synchronization

		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}
};

}
}
