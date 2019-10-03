#pragma once

#include "Codegen/Resources/ResourceAllocator.h"

#include "Codegen/Resources/RegisterResources.h"

#include "PTX/Type.h"
#include "PTX/Operands/Variables/Register.h"

#include "Utils/Logger.h"

namespace Codegen {

class RegisterAllocator : public ResourceAllocator<RegisterResources>
{
public:
	RegisterAllocator(RegisterAllocator *parent = nullptr) : m_parent(parent) {}

	template<class T>
	const PTX::Register<T> *AllocateRegister(const std::string& identifier, const PTX::Register<PTX::PredicateType> *predicate = nullptr)
	{
		return this->GetResources<T>()->AllocateRegister(identifier, predicate);
	}

	template<class T>
	const PTX::Register<T> *CompressRegister(const std::string& identifier, const PTX::Register<PTX::PredicateType> *predicate)
	{
		return this->GetResources<T>()->CompressRegister(identifier, predicate);
	}

	template<class T>
	void AddCompressedRegister(const std::string& identifier, const PTX::Register<T> *value, const PTX::Register<PTX::PredicateType> *predicate)
	{
		this->GetResources<T>()->AddCompressedRegister(identifier, value, predicate);
	}

	template<class T>
	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		if (ContainsKey<T>(identifier))
		{
			return this->GetResources<T>(false)->GetRegister(identifier);
		}
		if (m_parent != nullptr)
		{
			return m_parent->GetRegister<T>(identifier);
		}
		Utils::Logger::LogError("PTX::Register(" + identifier + ", " + T::Name() + ") not found");
	}

	template<class T>
	const PTX::Register<PTX::PredicateType> *GetCompressionRegister(const std::string& identifier) const
	{
		if (ContainsKey<T>(identifier))
		{
			return this->GetResources<T>(false)->GetCompressionRegister(identifier);
		}
		if (m_parent != nullptr)
		{
			return m_parent->GetCompressionRegister<T>(identifier);
		}
		Utils::Logger::LogError("PTX::Register(" + identifier + ", " + T::Name() + ") not found");
	}

	template<class T>
	const PTX::Register<T> *AllocateTemporary()
	{
		return this->GetResources<T>()->AllocateTemporary();
	}

	template<class T>
	const PTX::Register<T> *AllocateTemporary(const std::string& identifier)
	{
		auto resources = GetResources<T>();
		if (resources->ContainsTemporary(identifier))
		{
			return resources->GetTemporary(identifier);
		}
		return this->GetResources<T>()->AllocateTemporary(identifier);
	}

private:
	RegisterAllocator *m_parent = nullptr;
};

}
