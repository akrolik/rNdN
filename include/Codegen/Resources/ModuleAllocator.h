#pragma once

#include "Codegen/Resources/ResourceAllocator.h"

#include "Codegen/Resources/ModuleResources.h"

#include "PTX/Tree/Tree.h"

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
		if (m_dynamicSharedMemorySize > 0)
		{
			declarations.push_back(m_sharedMemoryDeclaration);
		}
		return declarations;
	}

	template<class T>
	const PTX::GlobalVariable<T> *AllocateGlobalVariable(const std::string& identifier)
	{
		return this->GetResources<T>()->AllocateGlobalVariable(identifier);
	}

	template<class T>
	bool ContainsGlobalVariable(const std::string& identifier) const
	{
		auto resources = this->GetResources<T>(false);
		if (resources != nullptr)
		{
			return resources->ContainsGlobalVariable(identifier);
		}
		return false;
	}

	template<class T>
	const PTX::GlobalVariable<T> *GetGlobalVariable(const std::string& identifier) const
	{
		if (ContainsGlobalVariable<T>(identifier))
		{
			return this->GetResources<T>(false)->GetGlobalVariable(identifier);
		}
		return false;
	}

	template<class T>
	const PTX::ConstVariable<T> *AllocateConstVariable(const std::string& identifier, const std::vector<typename T::SystemType>& initializer)
	{
		return this->GetResources<T>()->AllocateConstVariable(identifier, initializer);
	}

	template<class T>
	bool ContainsConstVariable(const std::string& identifier) const
	{
		auto resources = this->GetResources<T>(false);
		if (resources != nullptr)
		{
			return resources->ContainsConstVariable(identifier);
		}
		return false;
	}

	template<class T>
	const PTX::ConstVariable<T> *GetConstVariable(const std::string& identifier) const
	{
		if (ContainsConstVariable<T>(identifier))
		{
			return this->GetResources<T>(false)->GetConstVariable(identifier);
		}
		return false;
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

		auto bytes = sizeof(typename T::SystemType) * size;
		if (bytes > m_dynamicSharedMemorySize)
		{
			m_dynamicSharedMemorySize = bytes;
		}

		return sharedMemory;
	}

	unsigned int GetDynamicSharedMemorySize() const { return m_dynamicSharedMemorySize; }

	std::vector<const PTX::Declaration *> GetExternalDeclarations() const
	{
		std::vector<const PTX::Declaration *> vector;
		vector.insert(std::end(vector), std::begin(m_externalDeclarations), std::end(m_externalDeclarations));
		return vector;
	}

	template<class R>
	void AddExternalFunction(const PTX::FunctionDeclaration<R> *declaration)
	{
		if (m_externalDeclarations.find(declaration) == m_externalDeclarations.end())
		{
			m_externalDeclarations.insert(declaration);
		}
	}

private:
	PTX::TypedVariableDeclaration<PTX::ArrayType<PTX::Bit8Type, PTX::DynamicSize>, PTX::SharedSpace> *m_sharedMemoryDeclaration = nullptr;
	unsigned int m_dynamicSharedMemorySize = 0;

	std::set<const PTX::Declaration *> m_externalDeclarations;
};

}
