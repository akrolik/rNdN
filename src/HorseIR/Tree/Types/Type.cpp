#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Tree/Types/BasicType.h"
#include "HorseIR/Tree/Types/DictionaryType.h"
#include "HorseIR/Tree/Types/EnumerationType.h"
#include "HorseIR/Tree/Types/FunctionType.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/KeyedTableType.h"
#include "HorseIR/Tree/Types/TableType.h"

namespace HorseIR {

bool Type::operator==(const Type& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Basic:
				return static_cast<const BasicType&>(*this) == static_cast<const BasicType&>(other);
			case Kind::Function:
				return static_cast<const FunctionType&>(*this) == static_cast<const FunctionType&>(other);
			case Kind::List:
				return static_cast<const ListType&>(*this) == static_cast<const ListType&>(other);
			case Kind::Table:
				return static_cast<const TableType&>(*this) == static_cast<const TableType&>(other);
			case Kind::Dictionary:
				return static_cast<const DictionaryType&>(*this) == static_cast<const DictionaryType&>(other);
			case Kind::Enumeration:
				return static_cast<const EnumerationType&>(*this) == static_cast<const EnumerationType&>(other);
			case Kind::KeyedTable:
				return static_cast<const KeyedTableType&>(*this) == static_cast<const KeyedTableType&>(other);
		}
	}
	return false;
}

bool Type::operator!=(const Type& other) const
{
	return !(*this == other);
}

}
