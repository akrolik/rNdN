#pragma once

#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

#include "Frontend/Codegen/Resources/Resources.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<template<class> class R>
class ResourceAllocator
{
public:
	virtual std::vector<const PTX::VariableDeclaration *> GetDeclarations() const
	{
		std::vector<const PTX::VariableDeclaration *> declarations;
		for (const auto& resources : m_resourcesMap)
		{
			auto _declarations = resources.second->GetDeclarations();
			declarations.insert(declarations.end(), _declarations.begin(), _declarations.end());
		}
		return declarations;
	}

protected:
	template<class T>
	R<T> *GetResources(bool alloc = true) const
	{
		std::type_index key = typeid(T);
		if (m_resourcesMap.find(key) == m_resourcesMap.end())
		{
			if (!alloc)
			{
				return nullptr;
			}
			m_resourcesMap.insert({key, new R<T>()});
		}
		return static_cast<R<T> *>(m_resourcesMap.at(key));
	}

	mutable std::unordered_map<std::type_index, Resources *> m_resourcesMap;
};

}
}
