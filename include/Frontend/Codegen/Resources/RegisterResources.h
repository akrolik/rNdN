#pragma once

#include "Frontend/Codegen/Resources/Resources.h"

#include "PTX/Tree/Tree.h"

#include "Libraries/robin_hood.h"

namespace Frontend {
namespace Codegen {

enum class RegisterReductionGranularity {
	Single,
	Warp,
	Block
};

enum class RegisterReductionOperation {
	None,
	Add,
	Maximum,
	Minimum
};

template<class T>
class RegisterResources : public Resources
{
public:
	std::vector<PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return { m_declaration };
	}
	
	// Register declarations

	PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		if (m_registersMap.find(identifier) != m_registersMap.end())
		{
			return m_registersMap.at(identifier);
		}

		auto name = "%" + T::TypePrefix() + "_" + identifier;
		m_declaration->AddNames(name);

		auto resource = m_declaration->GetVariable(name);
		m_registersMap.insert({identifier, resource});

		return resource;
	}

	bool ContainsRegister(const std::string& identifier) const
	{
		return m_registersMap.find(identifier) != m_registersMap.end();
	}

	PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return m_registersMap.at(identifier);
	}

	// Temporary registers

	PTX::Register<T> *AllocateTemporary()
	{
		unsigned int temp = m_temporaries++;
		std::string name = "$" + T::TypePrefix();
		m_declaration->UpdateName(name, temp + 1);
		auto resource = m_declaration->GetVariable(name, temp);
		return resource;
	}

	// Compresed register flag

	void SetCompressedRegister(PTX::Register<T> *value, PTX::Register<PTX::PredicateType> *predicate)
	{
		m_compressedMap[value] = predicate;
	}

	bool IsCompressedRegister(PTX::Register<T> *value) const
	{
		return m_compressedMap.find(value) != m_compressedMap.end();
	}

	PTX::Register<PTX::PredicateType> *GetCompressedRegister(PTX::Register<T> *value) const
	{
		return m_compressedMap.at(value);
	}

	// Indexed register flag

	void SetIndexedRegister(PTX::Register<T> *value, PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		m_indexedMap[value] = index;
	}

	bool IsIndexedRegister(PTX::Register<T> *value) const
	{
		return m_indexedMap.find(value) != m_indexedMap.end();
	}

	PTX::TypedOperand<PTX::UInt32Type> *GetIndexedRegister(PTX::Register<T> *value) const
	{
		return m_indexedMap.at(value);
	}

	// Reduction register flag

	void SetReductionRegister(PTX::Register<T> *value, RegisterReductionGranularity granularity, RegisterReductionOperation op)
	{
		m_reductionMap[value] = {granularity, op};
	}

	bool IsReductionRegister(PTX::Register<T> *value) const
	{
		return m_reductionMap.find(value) != m_reductionMap.end();
	}

	std::pair<RegisterReductionGranularity, RegisterReductionOperation> GetReductionRegister(PTX::Register<T> *value) const
	{
		return m_reductionMap.at(value);
	}

private:
	PTX::RegisterDeclaration<T> *m_declaration = new PTX::RegisterDeclaration<T>();

	robin_hood::unordered_map<std::string, PTX::Register<T> *> m_registersMap;
	unsigned int m_temporaries = 0;

	robin_hood::unordered_map<PTX::Register<T> *, PTX::Register<PTX::PredicateType> *> m_compressedMap;
	robin_hood::unordered_map<PTX::Register<T> *, PTX::TypedOperand<PTX::UInt32Type> *> m_indexedMap;
	robin_hood::unordered_map<PTX::Register<T> *, std::pair<RegisterReductionGranularity, RegisterReductionOperation>> m_reductionMap;
};

}
}
