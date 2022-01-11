#pragma once

#include "Frontend/Codegen/Generators/Functions/FunctionGenerator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/NameUtils.h"
#include "Frontend/Codegen/Generators/Data/ParameterGenerator.h"
#include "Frontend/Codegen/Generators/Data/ParameterLoadGenerator.h"
#include "Frontend/Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadIndexGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class VectorFunctionGenerator : public FunctionGenerator<B>
{
public:
	using FunctionGenerator<B>::FunctionGenerator;

	std::string Name() const override { return "VectorFunctionGenerator"; }

	void Generate(const HorseIR::Function *function)
	{
		// Setup the parameter (in/out) declarations in the kernel

		ParameterGenerator<B> parameterGenerator(this->m_builder);
		parameterGenerator.Generate(function);

		// Check if the geometry size is dynamically specified
		
		auto& inputOptions = this->m_builder.GetInputOptions();
		const auto vectorGeometry = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::VectorShape>(inputOptions.ThreadGeometry);

		if (HorseIR::Analysis::ShapeUtils::IsDynamicSize(vectorGeometry->GetSize()))
		{
			// Load the geometry size from the input

			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::ThreadGeometryDataSize));

			auto geometryParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::ThreadGeometryDataSize);

			ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
			valueLoadGenerator.GenerateConstant(geometryParameter);
		}

		// Determine the block index

		if (inputOptions.InOrderBlocks)
		{
			// https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf

			auto resources = this->m_builder.GetLocalResources();

			// Initialize a global variable for counting the number of initialized blocks

			this->m_builder.AddStatement(new PTX::CommentStatement("g_initBlocks/s_blockIndex"));

			auto globalResources = this->m_builder.GetGlobalResources();
			auto g_initBlocks = globalResources->template AllocateGlobalVariable<PTX::UInt32Type>(this->m_builder.UniqueIdentifier("initBlocks"));

			// Setup a shared variable for the block storing the unique (in order) block index

			auto kernelResources = this->m_builder.GetKernelResources();
			auto s_blockIndex = kernelResources->template AllocateSharedVariable<PTX::UInt32Type>("blockIndex");

			// Only load/increment the global counter once per block (local index 0)

			ThreadIndexGenerator<B> indexGenerator(this->m_builder);
			auto localIndex = indexGenerator.GenerateLocalIndex();

			this->m_builder.AddIfStatement("BLOCK_INDEX", [&]()
			{
				auto local0Predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					local0Predicate, localIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual
				));
				return std::make_tuple(local0Predicate, false);
			},
			[&]()
			{
				// Atomically increment the global block index

				AddressGenerator<B, PTX::UInt32Type> addressGenerator(this->m_builder);
				auto g_initBlocksAddress = addressGenerator.GenerateAddress(g_initBlocks);

				auto blockIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
				this->m_builder.AddStatement(new PTX::AtomicInstruction<B, PTX::UInt32Type, PTX::GlobalSpace>(
					blockIndex, g_initBlocksAddress, new PTX::UInt32Value(1), PTX::UInt32Type::AtomicOperation::Add
				));

				// Store the unique block index for the entire block in shared member

				auto s_blockIndexAddress = addressGenerator.GenerateAddress(s_blockIndex);
				this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(s_blockIndexAddress, blockIndex));

				// Reset the block index (the kernel may be executed multiple times)

				this->m_builder.AddIfStatement("RESET", [&]()
				{
					auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
					auto nctaidxP1 = resources->template AllocateTemporary<PTX::UInt32Type>();

					SpecialRegisterGenerator specialGenerator(this->m_builder);
					auto nctaidx = specialGenerator.GenerateBlockCount();

					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(nctaidxP1, nctaidx, new PTX::UInt32Value(1)));
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
						predicate, blockIndex, nctaidx, PTX::UInt32Type::ComparisonOperator::NotEqual
					));

					return std::make_pair(predicate, false);
				},
				[&]()
				{
					auto value = resources->template AllocateTemporary<PTX::UInt32Type>();

					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(value, new PTX::UInt32Value(0)));
					this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::GlobalSpace>(g_initBlocksAddress, value));
				});
			});

			// Synchronize the block index across all threads in the block
			
			BarrierGenerator<B> barrierGenerator(this->m_builder);
			barrierGenerator.Generate();
		}

		// Initialize the parameters from the address space

		ParameterLoadGenerator<B> parameterLoadGenerator(this->m_builder);
		parameterLoadGenerator.Generate(function);

		// Generate the function body

		FunctionGenerator<B>::Visit(function);
	}

	void Visit(const HorseIR::ReturnStatement *returnS) override
	{
		// Finalize the vector function return statement by a return instruction

		FunctionGenerator<B>::Visit(returnS);
		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}
};

}
}
