#include "Runtime/DataObjects/DataList.h"

#include "HorseIR/Utils/TypeUtils.h"

#include "Utils/Logger.h"

namespace Runtime {

void DataList::AddElement(DataObject *element)
{
	// Check that this is either the same type as the list contents or that the list is heterogenous

	m_type->AddElementType(element->GetType()->Clone());
	m_elements.push_back(element);
}

std::string DataList::DebugDump() const
{
	std::string string = "[";
	for (const auto& cell : m_elements)
	{
		string += cell->DebugDump();
	}
	return (string + "]");
}

}
