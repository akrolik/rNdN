#pragma once

#include "Frontend/Codegen/Resources/ResourceAllocator.h"

#include "Frontend/Codegen/Resources/RegisterResources.h"

#include "PTX/Tree/Tree.h"

#include "Utils/Logger.h"

namespace Frontend {
namespace Codegen {

class RegisterAllocator : public ResourceAllocator<RegisterResources>
{
public:
	RegisterAllocator(RegisterAllocator *parent = nullptr) : m_parent(parent) {}

	// Registers

	template<class T>
	const PTX::Register<T> *AllocateRegister(const std::string& identifier)
	{
		return this->GetResources<T>()->AllocateRegister(identifier);
	}

	template<class T>
	bool ContainsRegister(const std::string& identifier) const
	{
		auto resources = GetResources<T>(false);
		if (resources != nullptr)
		{
			return resources->ContainsRegister(identifier);
		}
		return false;
	}

	template<class T>
	const PTX::Register<T> *GetRegister(const std::string& identifier) const
	{
		if (ContainsRegister<T>(identifier))
		{
			return this->GetResources<T>(false)->GetRegister(identifier);
		}
		if (m_parent != nullptr)
		{
			return m_parent->GetRegister<T>(identifier);
		}
		Utils::Logger::LogError("PTX::Register(" + identifier + ", " + T::Name() + ") not found");
	}

	// Temporary registers

	template<class T>
	const PTX::Register<T> *AllocateTemporary()
	{
		return this->GetResources<T>()->AllocateTemporary();
	}

	// Compresed register flag

	template<class T>
	void SetCompressedRegister(const PTX::Register<T> *value, const PTX::Register<PTX::PredicateType> *predicate)
	{
		this->GetResources<T>()->SetCompressedRegister(value, predicate);
	}

	template<class T>
	bool IsCompressedRegister(const PTX::Register<T> *value) const
	{
		if (auto resources = this->GetResources<T>(false))
		{
			if (resources->IsCompressedRegister(value))
			{
				return true;
			}
		}
		if (m_parent != nullptr)
		{
			return m_parent->IsCompressedRegister(value);
		}
		return false;
	}

	template<class T>
	const PTX::Register<PTX::PredicateType> *GetCompressedRegister(const PTX::Register<T> *value) const
	{
		if (auto resources = this->GetResources<T>(false))
		{
			if (resources->IsCompressedRegister(value))
			{
				return resources->GetCompressedRegister(value);
			}
		}
		if (m_parent != nullptr)
		{
			return m_parent->GetCompressedRegister(value);
		}
		return nullptr;
	}

	// Indexed data register flag

	template<class T>
	void SetIndexedRegister(const PTX::Register<T> *value, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		this->GetResources<T>()->SetIndexedRegister(value, index);
	}

	template<class T>
	bool IsIndexedRegister(const PTX::Register<T> *value) const
	{
		if (auto resources = this->GetResources<T>(false))
		{
			if (resources->IsIndexedRegister(value))
			{
				return true;
			}
		}
		if (m_parent != nullptr)
		{
			return m_parent->IsIndexedRegister(value);
		}
		return false;
	}

	template<class T>
	const PTX::TypedOperand<PTX::UInt32Type> *GetIndexedRegister(const PTX::Register<T> *value) const
	{
		if (auto resources = this->GetResources<T>(false))
		{
			if (resources->IsIndexedRegister(value))
			{
				return resources->GetIndexedRegister(value);
			}
		}
		if (m_parent != nullptr)
		{
			return m_parent->GetIndexedRegister(value);
		}
		return nullptr;
	}

	// Reduction register flag

	template<class T>
	void SetReductionRegister(const PTX::Register<T> *value, RegisterReductionGranularity granularity, RegisterReductionOperation op)
	{
		this->GetResources<T>()->SetReductionRegister(value, granularity, op);
	}

	template<class T>
	bool IsReductionRegister(const PTX::Register<T> *value) const
	{
		if (auto resources = this->GetResources<T>(false))
		{
			if (resources->IsReductionRegister(value))
			{
				return true;
			}
		}
		if (m_parent != nullptr)
		{
			return m_parent->IsReductionRegister(value);
		}
		return false;
	}

	template<class T>
	std::pair<RegisterReductionGranularity, RegisterReductionOperation> GetReductionRegister(const PTX::Register<T> *value) const
	{
		if (auto resources = this->GetResources<T>(false))
		{
			if (resources->IsReductionRegister(value))
			{
				return resources->GetReductionRegister(value);
			}
		}
		if (m_parent != nullptr)
		{
			return m_parent->GetReductionRegister(value);
		}
		Utils::Logger::LogError("Reduction register not found");
	}

private:
	RegisterAllocator *m_parent = nullptr;
};

}
}
