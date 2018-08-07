#include "Runtime/DataObjects/DataList.h"

#include "Utils/Logger.h"

namespace Runtime {

void DataList::AddElement(DataVector *element)
{
	auto type = m_type->GetElementType();
	if (type->GetKind() != HorseIR::Type::Kind::Basic || static_cast<const HorseIR::BasicType *>(type)->GetKind() != HorseIR::BasicType::Kind::Wildcard)
	{
		if (*type != *element->GetType())
		{
			Utils::Logger::LogError("Cannot add element of type " + element->GetType()->ToString() + " to list of type " + m_type->ToString());
		}
	}
	m_elements.push_back(element);
}

void DataList::Dump() const
{
	//TODO:
}

}
