#pragma once

#include <map>

#include "HorseIR/Traversal/ForwardTraversal.h"
#include "HorseIR/Tree/Method.h"

class ResourceAllocator : public HorseIR::ForwardTraversal
{
public:
	void AllocateResources(HorseIR::Method *method)
	{
		m_map.clear();
		method->Accept(*this);
	}

	unsigned int GetResource(std::string identifier) const
	{
		return m_map.at(identifier);
	}

	void Visit(HorseIR::Method *method) override
	{
		//TODO: parameters allocation
		HorseIR::ForwardTraversal::Visit(method);
	}

	void Visit(HorseIR::AssignStatement *assign) override
	{
		m_map[assign->GetIdentifier()] = c++;
		HorseIR::ForwardTraversal::Visit(assign);
	}

private:
	std::map<std::string, unsigned int> m_map;
	unsigned int c = 0;
};
