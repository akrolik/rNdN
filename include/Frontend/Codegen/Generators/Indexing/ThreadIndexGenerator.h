#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/SpecialRegisterGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadGeometryGenerator.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class ThreadIndexGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "ThreadIndexGenerator"; }

	enum class Kind {
		Null,
		Lane,
		Warp,
		Local,
		Block,
		Global
	};

	PTX::TypedOperand<PTX::UInt32Type> *GenerateIndex(Kind kind)
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

	PTX::Register<PTX::UInt32Type> *GenerateLaneIndex()
	{
		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto tidx = specialGenerator.GenerateThreadIndex();

		auto resources = this->m_builder.GetLocalResources();
		auto laneid = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(laneid, tidx, PTX::SpecialConstant_WARP_SZ));

		return laneid;
	}

	PTX::Register<PTX::UInt32Type> *GenerateWarpIndex()
	{
		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto tidx = specialGenerator.GenerateThreadIndex();

		auto resources = this->m_builder.GetLocalResources();
		auto warpid = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(warpid, tidx, PTX::SpecialConstant_WARP_SZ));

		return warpid;
	}

	PTX::Register<PTX::UInt32Type> *GenerateLocalIndex()
	{
		SpecialRegisterGenerator specialGenerator(this->m_builder);
		return specialGenerator.GenerateThreadIndex();
	}

	PTX::Register<PTX::UInt32Type> *GenerateBlockIndex()
	{
		// Check if we use in order blocks (allocation happens at the beginning of execution), or using the system value

		auto& inputOptions = this->m_builder.GetInputOptions();
		if (inputOptions.InOrderBlocks)
		{
			// Load the special block index from shard memory

			auto kernelResources = this->m_builder.GetKernelResources();
			auto s_blockIndex = kernelResources->template GetSharedVariable<PTX::UInt32Type>("blockIndex");

			AddressGenerator<B, PTX::UInt32Type> addressGenerator(this->m_builder);
			auto s_blockIndexAddress = addressGenerator.GenerateAddress(s_blockIndex);

			auto resources = this->m_builder.GetLocalResources();
			auto blockIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::UInt32Type, PTX::SharedSpace>(blockIndex, s_blockIndexAddress));

			return blockIndex;
		}
		else
		{
			SpecialRegisterGenerator specialGenerator(this->m_builder);
			return specialGenerator.GenerateBlockIndex();
		}
	}

	PTX::Register<PTX::UInt32Type> *GenerateGlobalIndex()
	{
		if (auto index = this->m_builder.GetThreadIndex())
		{
			auto resources = this->m_builder.GetLocalResources();
			auto temp = resources->template AllocateTemporary<PTX::UInt32Type>();

			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(temp, index));

			return temp;
		}

		// Compute the thread index as blockSize * blockIndex + threadIndex

		auto tidx = GenerateLocalIndex();
		auto ctaidx = GenerateBlockIndex();

		SpecialRegisterGenerator specialGenerator(this->m_builder);
		auto ntidx = specialGenerator.GenerateThreadCount();

		auto resources = this->m_builder.GetLocalResources();
		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::MADInstruction<PTX::UInt32Type>(
			index, ctaidx, ntidx, tidx, PTX::HalfModifier<PTX::UInt32Type>::Half::Lower
		));
		this->m_builder.SetThreadIndex(index);

		return index;
	}

	PTX::Register<PTX::UInt32Type> *GenerateListLocalIndex()
	{
		// List local index = localIndex % cellThreads

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto listThreads = geometryGenerator.GenerateListThreads();
		auto localIndex = GenerateLocalIndex();

		auto resources = this->m_builder.GetLocalResources();
		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(index, localIndex, listThreads));

		return index;
	}
};

}
}
