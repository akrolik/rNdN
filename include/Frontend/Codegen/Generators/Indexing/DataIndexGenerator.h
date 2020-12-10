#pragma once

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/NameUtils.h"
#include "Frontend/Codegen/Generators/Indexing/SpecialRegisterGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadGeometryGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/ThreadIndexGenerator.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class DataIndexGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "DataIndexGenerator"; }

	enum class Kind {
		Broadcast,
		VectorData,
		ListBroadcast,
		ListData
	};

	PTX::TypedOperand<PTX::UInt32Type> *GenerateIndex(Kind indexKind)
	{
		switch (indexKind)
		{
			case Kind::Broadcast:
				return new PTX::UInt32Value(0);
			case Kind::VectorData:
				return GenerateVectorIndex();
			case Kind::ListBroadcast:
				return GenerateListIndex();
			case Kind::ListData:
				return GenerateListDataIndex();
		}

		return nullptr;
	}

	PTX::Register<PTX::UInt32Type> *GenerateDataIndex()
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (inputOptions.IsVectorGeometry())
		{
			return GenerateVectorIndex();
		}
		else if (inputOptions.IsListGeometry())
		{
			return GenerateListDataIndex();
		}
		Error("data index for geometry");
	}

	PTX::Register<PTX::UInt32Type> *GenerateVectorIndex()
	{
		ThreadIndexGenerator<B> threadGenerator(this->m_builder);
		return threadGenerator.GenerateGlobalIndex();
	}

	PTX::Register<PTX::UInt32Type> *GenerateListIndex()
	{
		// List index = globalIndex / listThreads

		ThreadIndexGenerator<B> threadGenerator(this->m_builder);
		auto globalIndex = threadGenerator.GenerateGlobalIndex();

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto listThreads = geometryGenerator.GenerateListThreads();

		auto resources = this->m_builder.GetLocalResources();
		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(index, globalIndex, listThreads));

		return index;
	}

	PTX::Register<PTX::UInt32Type> *GenerateListDataIndex()
	{
		// Data index = globalIndex % cellThreads

		auto resources = this->m_builder.GetLocalResources();

		if (resources->ContainsRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryListDataIndex))
		{
			// Fetch the special register which is computed for each thread and updated by the loop

			return resources->GetRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryListDataIndex);
		}

		ThreadIndexGenerator<B> threadGenerator(this->m_builder);
		auto globalIndex = threadGenerator.GenerateGlobalIndex();

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto listThreads = geometryGenerator.GenerateListThreads();

		auto index = resources->AllocateRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryListDataIndex);

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(index, globalIndex, listThreads));

		return index;
	}
};

}
}
