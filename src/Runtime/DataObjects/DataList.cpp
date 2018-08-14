#include "Runtime/DataObjects/DataList.h"

#include "HorseIR/TypeUtils.h"

#include "Utils/Logger.h"

namespace Runtime {

void DataList::AddElement(DataVector *element)
{
	// Check that this is either the same type as the list contents or that the list is heterogenous

	auto elementType = m_type->GetElementType();
	if (auto basicType = HorseIR::GetType<HorseIR::BasicType>(elementType); basicType == nullptr || basicType->GetKind() != HorseIR::BasicType::Kind::Wildcard)
	{
		if (*elementType != *element->GetType())
		{
			Utils::Logger::LogError("Cannot add element of type '" + element->GetType()->ToString() + "' to list of type '" + m_type->ToString() + "'");
		}
	}
	m_elements.push_back(element);
}

void DataList::Dump() const
{
	//TODO:
}

}
