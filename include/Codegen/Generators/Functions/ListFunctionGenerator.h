#pragma once

#include "Codegen/Generators/Functions/FunctionGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/InputOptions.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/GeometryGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/Data/ParameterGenerator.h"
#include "Codegen/Generators/Data/ParameterLoadGenerator.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ListFunctionGenerator : public FunctionGenerator<B>
{
public:
	using FunctionGenerator<B>::FunctionGenerator;

	void Generate(const HorseIR::Function *function)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto& inputOptions = this->m_builder.GetInputOptions();
		auto& kernelOptions = this->m_builder.GetKernelOptions();

		ParameterGenerator<B> parameterGenerator(this->m_builder);
		ParameterLoadGenerator<B> parameterLoadGenerator(this->m_builder);
		ValueLoadGenerator<B> valueLoadGenerator(this->m_builder);

		// Setup the thread count for each cell

		if (inputOptions.ListCellThreads == InputOptions::DynamicSize)
		{
			// Load the dynamic thread allocation parameter

			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::GeometryCellThreads));

			auto parameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::GeometryCellThreads);
			valueLoadGenerator.template GenerateConstant<PTX::UInt32Type>(parameter);
		}
		else
		{
			// Restrict the number of threads in the cell block if we have already specified a size

			kernelOptions.SetBlockSize(inputOptions.ListCellThreads);
		}

		// Load the number of cells from the input if required

		const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry);
		if (Analysis::ShapeUtils::IsDynamicSize(listGeometry->GetListSize()))
		{
			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::GeometryListSize));

			auto parameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::GeometryListSize);
			valueLoadGenerator.template GenerateConstant<PTX::UInt32Type>(parameter);
		}

		// Check of the cell is within bounds

		this->m_builder.AddStatement(new PTX::CommentStatement("bounds check"));

		IndexGenerator indexGenerator(this->m_builder);
		auto cellIndex = indexGenerator.GenerateCellIndex();

		GeometryGenerator geometryGenerator(this->m_builder);
		auto cellCount = geometryGenerator.GenerateListSize();

		auto exitPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(exitPredicate, cellIndex, cellCount, PTX::UInt32Type::ComparisonOperator::GreaterEqual));

		auto exitInstruction = new PTX::ReturnInstruction();
		exitInstruction->SetPredicate(exitPredicate);
		this->m_builder.AddStatement(exitInstruction);

		// Load the geometry cell sizes

		this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::GeometryDataSize));

		auto geometryDataSizeParameter = parameterGenerator.template GeneratePointer<PTX::UInt32Type>(NameUtils::GeometryDataSize);
		parameterLoadGenerator.template GenerateVector<PTX::UInt32Type>(geometryDataSizeParameter);
		valueLoadGenerator.template GeneratePointer<PTX::UInt32Type>(geometryDataSizeParameter, cellIndex);

		// Initialize the special local index register

		this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::GeometryThreadIndex));

		auto index = indexGenerator.GenerateInitialCellDataIndex();

		// Fetch the arguments from the .param address space

		for (const auto& parameter : function->GetParameters())
		{
			parameterLoadGenerator.Generate(parameter);
		}
		parameterLoadGenerator.Generate(function->GetReturnTypes());

		// Generate the starting index for the loop

		this->m_builder.AddStatement(new PTX::CommentStatement("loop init"));

		// Generate the loop bound

		auto cellSize = geometryGenerator.GenerateCellSize();
		auto bound = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(bound, index, cellSize));

		// Construct the standard loop structure
		//
		//   START:
		//      setp %p, ...
		//      @%p br END
		//
		//      <load>
		//      <body>
		//      <increment>
		//
		//      br START
		//
		//   END:

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(startLabel);
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, bound, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Construct the loop body according to the standard function generator

		for (const auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
		}

		// Increment the thread index by the number of threads per cell

		this->m_builder.AddStatement(new PTX::CommentStatement("loop post"));

		auto cellThreads = indexGenerator.GenerateCellThreads();
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(index, index, cellThreads));

		// Complete the loop structure

		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));
		this->m_builder.AddStatement(new PTX::BlankStatement());
		this->m_builder.AddStatement(endLabel);

		// Lastly, end with a return instruction - this only generated at the end for body synchronization

		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}
};

}
