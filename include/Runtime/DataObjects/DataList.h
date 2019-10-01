#pragma once

#include "Runtime/DataObjects/DataObject.h"

#include <string>
#include <vector>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Runtime {

class DataList : public DataObject
{
public:
	DataList(DataObject *element) : DataList(std::vector<DataObject *>({element})) {}
	DataList(const std::vector<DataObject *>& elements) : m_elements(elements)
	{
		std::vector<HorseIR::Type *> elementTypes;
		for (const auto& element : elements)
		{
			elementTypes.push_back(element->GetType()->Clone());
		}
		m_type = new HorseIR::ListType(elementTypes);
	}

	const HorseIR::ListType *GetType() const { return m_type; }

	void AddElement(DataObject *element);
	DataObject *GetElement(unsigned int index) { return m_elements.at(index); }
	size_t GetElementCount() const { return m_elements.size(); }

	std::string Description() const override
	{
		std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
		bool first = true;
		for (const auto& object : m_elements)
		{
			if (!first)
			{
				description += ", ";
			}
			first = false;
			description += object->Description();
		}
		return description + "}";
	}

	std::string DebugDump() const override;

private:
	HorseIR::ListType *m_type = nullptr;

	std::vector<DataObject *> m_elements;
};

}
