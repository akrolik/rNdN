#pragma once

#include "Codegen/Resources/Resources.h"

#include "PTX/PTX.h"

namespace Codegen {

template<class T>
class RegisterResources : public Resources
{
public:
	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return { m_declaration };
	}

	const PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		if (m_registersMap.find(identifier) != m_registersMap.end())
		{
			return m_registersMap.at(identifier);
		}
		else
		{
			auto name = "%" + T::TypePrefix() + "_" + identifier;
			m_declaration->AddNames(name);
			const auto resource = m_declaration->GetVariable(name);
			m_registersMap.insert({identifier, resource});
			return resource;
		}
	}

	const PTX::Register<T> *AllocateTemporary()
	{
		unsigned int temp = m_temporaries++;
		std::string name = "$" + T::TypePrefix();
		m_declaration->UpdateName(name, temp + 1);
		const auto resource = m_declaration->GetVariable(name, temp);
		return resource;
	}

	bool ContainsKey(const std::string& identifier) const override
	{
		return m_registersMap.find(identifier) != m_registersMap.end();
	}

	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return m_registersMap.at(identifier);
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

	enum class ReductionGranularity {
		Warp,
		Block
	};

	enum class ReductionOperation {
		Add,
		Maximum,
		Minimum
	};

	void SetReductionRegister(const PTX::Register<T> *value, ReductionGranularity granularity, ReductionOperation op)
	{
		m_reductionMap[value] = {granularity, op};
	}

	bool IsReductionRegister(const PTX::Register<T> *value) const
	{
		return m_reductionMap.find(value) != m_reductionMap.end();
	}

	std::pair<ReductionGranularity, ReductionOperation> GetReductionRegister(const PTX::Register<T> *value) const
	{
		return m_reductionMap.at(value);
	}

private:
	PTX::RegisterDeclaration<T> *m_declaration = new PTX::RegisterDeclaration<T>();

	std::unordered_map<std::string, const PTX::Register<T> *> m_registersMap;
	unsigned int m_temporaries = 0;

	std::unordered_map<const PTX::Register<T> *, const PTX::Register<PTX::PredicateType> *> m_compressedMap;
	std::unordered_map<const PTX::Register<T> *, const PTX::TypedOperand<PTX::UInt32Type> *> m_indexedMap;
	std::unordered_map<const PTX::Register<T> *, std::pair<ReductionGranularity, ReductionOperation>> m_reductionMap;
};

}
