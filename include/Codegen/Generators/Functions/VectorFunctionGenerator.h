#pragma once

#include "Codegen/Generators/Functions/FunctionGenerator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/Data/ParameterGenerator.h"
#include "Codegen/Generators/Data/ParameterLoadGenerator.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class VectorFunctionGenerator : public FunctionGenerator<B>
{
public:
	using FunctionGenerator<B>::FunctionGenerator;

	void Generate(const HorseIR::Function *function)
	{
		// Initialize the parameters from the address space

		ParameterLoadGenerator<B> parameterLoadGenerator(this->m_builder);
		for (const auto& parameter : function->GetParameters())
		{
			parameterLoadGenerator.Generate(parameter);
		}
		parameterLoadGenerator.Generate(function->GetReturnTypes());

		auto& inputOptions = this->m_builder.GetInputOptions();

		// Check if the geometry size is dynamically specified
		
		const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry);
		if (Analysis::ShapeUtils::IsDynamicSize(vectorGeometry->GetSize()))
		{
			// Load the geometry size from the input

			this->m_builder.AddStatement(new PTX::CommentStatement(NameUtils::GeometryDataSize));

			ParameterGenerator<B> parameterGenerator(this->m_builder);
			auto geometryParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::GeometryDataSize);

			ValueLoadGenerator<B> valueLoadGenerator(this->m_builder);
			valueLoadGenerator.template GenerateConstant<PTX::UInt32Type>(geometryParameter);
		}

		if (inputOptions.InOrderBlocks)
		{
			auto resources = this->m_builder.GetLocalResources();

			this->m_builder.AddStatement(new PTX::CommentStatement("g_initBlocks/s_blockIndex"));

			// Initialize a global variable for counting the number of initialized blocks

			auto globalResources = this->m_builder.GetGlobalResources();
			auto g_initBlocks = globalResources->template AllocateGlobalVariable<PTX::UInt32Type>(this->m_builder.UniqueIdentifier("initBlocks"));

			// Setup a shared variable for the block storing the unique (in order) block index

			auto kernelResources = this->m_builder.GetKernelResources();
			auto s_blockIndex = kernelResources->template AllocateSharedVariable<PTX::UInt32Type>("blockIndex");

			// Only load/increment the global counter once per block (local index 0)

			IndexGenerator indexGenerator(this->m_builder);
			auto localIndex = indexGenerator.GenerateLocalIndex();

			auto local0Predicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				local0Predicate, localIndex, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual
			));

			auto blockIndexLabel = this->m_builder.CreateLabel("BLOCK_INDEX");
			this->m_builder.AddStatement(new PTX::BranchInstruction(blockIndexLabel, local0Predicate));

			// Atomically increment the global block index

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto g_initBlocksAddress = addressGenerator.GenerateAddress(g_initBlocks);

			auto blockIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::AtomicInstruction<B, PTX::UInt32Type, PTX::GlobalSpace, PTX::UInt32Type::AtomicOperation::Add>(
				blockIndex, g_initBlocksAddress, new PTX::UInt32Value(1)
			));

			// Compute the block index, keeping it within range (the kernel may be executed multiple times)

			SpecialRegisterGenerator specialGenerator(this->m_builder);
			auto nctaidx = specialGenerator.GenerateBlockCount();

			this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(blockIndex, blockIndex, nctaidx));

			// Store the unique block index for the entire block in shared member

			auto s_blockIndexAddress = addressGenerator.GenerateAddress(s_blockIndex);
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(s_blockIndexAddress, blockIndex));

			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(blockIndexLabel);

			// Synchronize the block index across all threads in the block
			
			BarrierGenerator barrierGenerator(this->m_builder);
			barrierGenerator.Generate();
		}

		for (const auto& statement : function->GetStatements())
		{
			statement->Accept(*this);
		}
	}

	void Visit(const HorseIR::ReturnStatement *returnS) override
	{
		// Finalize the vector function return statement by a return instruction

		FunctionGenerator<B>::Visit(returnS);
		this->m_builder.AddStatement(new PTX::ReturnInstruction());
	}
};

}
