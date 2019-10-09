#pragma once

#include "Codegen/Resources/Resources.h"

#include "PTX/PTX.h"

namespace Codegen {

template<class T>
class RegisterResources : public Resources
{
public:
	using ResourceType = std::pair<const PTX::Register<T> *, const PTX::Register<PTX::PredicateType> *>;

	std::vector<const PTX::VariableDeclaration *> GetDeclarations() const override
	{
		return { m_declaration };
	}

	//TODO: Move and not copy make_pair

	const PTX::Register<T> *AllocateRegister(const std::string& identifier, const PTX::Register<PTX::PredicateType> *predicate = nullptr)
	{
		auto name = "%" + T::TypePrefix() + "_" + identifier;
		m_declaration->AddNames(name);
		const auto resource = m_declaration->GetVariable(name);
		m_registersMap[identifier] = std::make_pair(resource, predicate);
		return resource;
	}

	const PTX::Register<T> *CompressRegister(const std::string& identifier, const PTX::Register<PTX::PredicateType> *predicate)
	{
		auto resource = GetRegister(identifier);
		m_registersMap[identifier] = std::make_pair(resource, predicate);
		return resource;
	}

	void AddCompressedRegister(const std::string& identifier, const PTX::Register<T> *value, const PTX::Register<PTX::PredicateType> *predicate)
	{
		m_registersMap[identifier] = std::make_pair(value, predicate);
	}

	bool ContainsKey(const std::string& identifier) const override
	{
		return m_registersMap.find(identifier) != m_registersMap.end();
	}

	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		return std::get<0>(m_registersMap.at(identifier));
	}

	const PTX::Register<PTX::PredicateType> *GetCompressionRegister(const std::string& identifier) const
	{
		return std::get<1>(m_registersMap.at(identifier));
	}

	const PTX::Register<T> *AllocateTemporary()
	{
		unsigned int temp = m_temporaries++;
		std::string name = "$" + T::TypePrefix();
		m_declaration->UpdateName(name, temp + 1);
		const auto resource = m_declaration->GetVariable(name, temp);
		return resource;
	}

	const PTX::Register<T> *AllocateTemporary(const std::string& identifier)
	{
		auto name = "$" + T::TypePrefix() + "_" + identifier;
		m_declaration->AddNames(name);
		const auto resource = m_declaration->GetVariable(name);
		m_temporariesMap[identifier] = resource;
		return resource;
	}

	bool ContainsTemporary(const std::string& identifier) const
	{
		return m_temporariesMap.find(identifier) != m_temporariesMap.end();
	}

	const PTX::Register<T> *GetTemporary(const std::string& identifier) const
	{
		return m_temporariesMap.at(identifier);
	}

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

	std::unordered_map<std::string, ResourceType> m_registersMap;
	std::unordered_map<std::string, const PTX::Register<T> *> m_temporariesMap;
	unsigned int m_temporaries = 0;

	std::unordered_map<const PTX::Register<T> *, std::pair<ReductionGranularity, ReductionOperation>> m_reductionMap;
};

}
