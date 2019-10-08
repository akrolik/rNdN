#pragma once

#include "Codegen/Resources/ResourceAllocator.h"

#include "Codegen/Resources/ModuleResources.h"

#include "PTX/PTX.h"

namespace Codegen {

class ModuleAllocator : public ResourceAllocator<ModuleResources>
{
public:
	constexpr static const char *DynamicVariableName = "$sdata";

	ModuleAllocator()
	{
		m_sharedMemoryDeclaration = new PTX::TypedVariableDeclaration<PTX::ArrayType<PTX::Bit8Type, PTX::DynamicSize>, PTX::SharedSpace>(DynamicVariableName);
		m_sharedMemoryDeclaration->SetLinkDirective(PTX::Declaration::LinkDirective::External);
	}

	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const
	{
		auto declarations = ResourceAllocator<ModuleResources>::GetDeclarations();
		if (m_sharedMemorySize > 0)
		{
			declarations.push_back(m_sharedMemoryDeclaration);
		}
		return declarations;
	}

	template<class T>
	const PTX::SharedVariable<T> *AllocateDynamicSharedMemory(unsigned int size)
	{
		auto alignment = m_sharedMemoryDeclaration->GetAlignment();
		auto typeAlignment = PTX::BitSize<T::TypeBits>::NumBytes;
		if (typeAlignment > alignment)
		{
			m_sharedMemoryDeclaration->SetAlignment(typeAlignment);
		}

		auto sharedMemoryBits = new PTX::ArrayVariableAdapter<PTX::Bit8Type, PTX::DynamicSize, PTX::SharedSpace>(m_sharedMemoryDeclaration->GetVariable(DynamicVariableName));
		auto sharedMemory = new PTX::VariableAdapter<T, PTX::Bit8Type, PTX::SharedSpace>(sharedMemoryBits);

		m_sharedMemorySize += sizeof(typename T::SystemType) * size;

		return sharedMemory;
	}

	template<class T>
	const PTX::SharedVariable<T> *AllocateSharedMemory()
	{
		return this->GetResources<T>()->AllocateSharedMemory();
	}

	unsigned int GetSharedMemorySize() const { return m_sharedMemorySize; }

	template<class R>
	void AddExternalFunction(PTX::FunctionDeclaration<R> *declaration)
	{
		if (m_externalDeclarations.find(declaration) == m_externalDeclarations.end())
		{
			m_externalDeclarations.insert(declaration);
		}
	}

private:
	PTX::TypedVariableDeclaration<PTX::ArrayType<PTX::Bit8Type, PTX::DynamicSize>, PTX::SharedSpace> *m_sharedMemoryDeclaration = nullptr;
	unsigned int m_sharedMemorySize = 0;

	std::set<PTX::Declaration *> m_externalDeclarations;
};

}
