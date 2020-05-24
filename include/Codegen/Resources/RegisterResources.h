#pragma once

#include "Codegen/Resources/Resources.h"

#include "PTX/PTX.h"

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
	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return { m_declaration };
	}
	
	// Register declarations

	const PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		if (m_registersMap.find(identifier) != m_registersMap.end())
		{
			return m_registersMap.at(identifier);
		}

		auto name = "%" + T::TypePrefix() + "_" + identifier;
		m_declaration->AddNames(name);

		const auto resource = m_declaration->GetVariable(name);
		m_registersMap.insert({identifier, resource});

		return resource;
	}

	bool ContainsRegister(const std::string& identifier) const
	{
		return m_registersMap.find(identifier) != m_registersMap.end();
	}

	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return m_registersMap.at(identifier);
	}

	// Temporary registers

	const PTX::Register<T> *AllocateTemporary()
	{
		unsigned int temp = m_temporaries++;
		std::string name = "$" + T::TypePrefix();
		m_declaration->UpdateName(name, temp + 1);
		const auto resource = m_declaration->GetVariable(name, temp);
		return resource;
	}

	// Compresed register flag

	void SetCompressedRegister(const PTX::Register<T> *value, const PTX::Register<PTX::PredicateType> *predicate)
	{
		m_compressedMap[value] = predicate;
	}

	bool IsCompressedRegister(const PTX::Register<T> *value) const
	{
		return m_compressedMap.find(value) != m_compressedMap.end();
	}

	const PTX::Register<PTX::PredicateType> *GetCompressedRegister(const PTX::Register<T> *value) const
	{
		return m_compressedMap.at(value);
	}

	// Indexed register flag

	void SetIndexedRegister(const PTX::Register<T> *value, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		m_indexedMap[value] = index;
	}

	bool IsIndexedRegister(const PTX::Register<T> *value) const
	{
		return m_indexedMap.find(value) != m_indexedMap.end();
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GetIndexedRegister(const PTX::Register<T> *value) const
	{
		return m_indexedMap.at(value);
	}

	// Reduction register flag

	void SetReductionRegister(const PTX::Register<T> *value, RegisterReductionGranularity granularity, RegisterReductionOperation op)
	{
		m_reductionMap[value] = {granularity, op};
	}

	bool IsReductionRegister(const PTX::Register<T> *value) const
	{
		return m_reductionMap.find(value) != m_reductionMap.end();
	}

	std::pair<RegisterReductionGranularity, RegisterReductionOperation> GetReductionRegister(const PTX::Register<T> *value) const
	{
		return m_reductionMap.at(value);
	}

private:
	PTX::RegisterDeclaration<T> *m_declaration = new PTX::RegisterDeclaration<T>();

	std::unordered_map<std::string, const PTX::Register<T> *> m_registersMap;
	unsigned int m_temporaries = 0;

	std::unordered_map<const PTX::Register<T> *, const PTX::Register<PTX::PredicateType> *> m_compressedMap;
	std::unordered_map<const PTX::Register<T> *, const PTX::TypedOperand<PTX::UInt32Type> *> m_indexedMap;
	std::unordered_map<const PTX::Register<T> *, std::pair<RegisterReductionGranularity, RegisterReductionOperation>> m_reductionMap;
};

}
